from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import logging
import os
from tqdm import tqdm

import typing
from typing import List, Text, Any, Optional, Dict

from rasa_nlu_gao.classifiers import INTENT_RANKING_LENGTH
from rasa_nlu_gao.components import Component
from multiprocessing import cpu_count
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import tensorflow as tf
    from rasa_nlu_gao.config import RasaNLUModelConfig
    from rasa_nlu_gao.training_data import TrainingData
    from rasa_nlu_gao.model import Metadata
    from rasa_nlu_gao.training_data import Message

try:
    import tensorflow as tf
except ImportError:
    tf = None

import argparse
import pickle
from os import path

from tensorflow.python.keras.utils import to_categorical
from rasa_nlu_gao.models.nlp_architect.callbacks import ConllCallback
from rasa_nlu_gao.models.nlp_architect.intent_extraction import MultiTaskIntentModel
from rasa_nlu_gao.models.nlp_architect.utils.embedding import get_embedding_matrix, load_word_embeddings
from rasa_nlu_gao.models.nlp_architect.utils.generic import one_hot
from rasa_nlu_gao.models.nlp_architect.utils.io import validate, validate_existing_directory, \
    validate_existing_filepath, validate_parent_exists
from rasa_nlu_gao.models.nlp_architect.utils.metrics import get_conll_scores
from rasa_nlu_gao.models.nlp_architect.intent_datasets import IntentDataset
from rasa_nlu_gao.models.nlp_architect.utils.generic import pad_sentences
from rasa_nlu_gao.models.nlp_architect.utils.text import SpacyInstance, bio_to_spans

class MultiTaskIntent(Component):
    """Intent classifier using supervised bert embeddings."""

    name = "multi_task_intent"

    provides = ["intent", "intent_ranking"]

    requires = ["tokens"]

    defaults = {
        # nn architecture
        "batch_size": 256,
        "epochs": 300,

        "learning_rate": 0.001,
        "sentence_length": 30,
        "token_emb_size": 100,
        "intent_hidden_size": 100,
        "lstm_hidden_size": 150,
        "tagger_dropout": 0.5,
        "embedding_model": None,
        
        "use_cudnn": False,

        # flag if tokenize intents
        "intent_tokenization_flag": False,
        "intent_split_symbol": '_',

        # visualization of accuracy
        "evaluate_every_num_epochs": 10,  # small values may hurt performance
        "evaluate_on_num_examples": 1000,  # large values may hurt performance

        "config_proto": {
            "device_count": cpu_count(),
            "inter_op_parallelism_threads": 0,
            "intra_op_parallelism_threads": 0,
            "allow_growth": True
        }
    }

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["tensorflow"]

    def _load_nn_architecture_params(self):
        self.batch_size = self.component_config['batch_size']
        self.epochs = self.component_config['epochs']

        self.learning_rate = self.component_config['learning_rate']
        self.sentence_length = self.component_config['sentence_length']
        self.token_emb_size = self.component_config['token_emb_size']
        self.intent_hidden_size = self.component_config['intent_hidden_size']
        self.lstm_hidden_size = self.component_config['lstm_hidden_size']
        self.tagger_dropout = self.component_config['tagger_dropout']
        self.embedding_model = self.component_config['embedding_model']
        self.use_cudnn = self.component_config['use_cudnn']

    def _load_flag_if_tokenize_intents(self):
        self.intent_tokenization_flag = self.component_config['intent_tokenization_flag']
        self.intent_split_symbol = self.component_config['intent_split_symbol']
        if self.intent_tokenization_flag and not self.intent_split_symbol:
            logger.warning("intent_split_symbol was not specified, "
                           "so intent tokenization will be ignored")
            self.intent_tokenization_flag = False

    def _load_visual_params(self):
        self.evaluate_every_num_epochs = self.component_config[
                                            'evaluate_every_num_epochs']
        if self.evaluate_every_num_epochs < 1:
            self.evaluate_every_num_epochs = self.epochs
        self.evaluate_on_num_examples = self.component_config[
                                            'evaluate_on_num_examples']

    @staticmethod
    def _check_tensorflow():
        if tf is None:
            raise ImportError(
                'Failed to import `tensorflow`. '
                'Please install `tensorflow`. '
                'For example with `pip install tensorflow`.')

    def __init__(self,
                 component_config=None,  # type: Optional[Dict[Text, Any]]
                 session=None,  # type: Optional[tf.Session]
                 graph=None  # type: Optional[tf.Graph]
                 ):
        # type: (...) -> None
        """Declare instant variables with default values"""
        self._check_tensorflow()
        super(MultiTaskIntent, self).__init__(component_config)

        # nn architecture parameters
        self._load_nn_architecture_params()

        # flag if tokenize intents
        self._load_flag_if_tokenize_intents()
        # visualization of accuracy
        self._load_visual_params()

        # tf related instances
        self.session = session
        self.graph = graph

    @staticmethod
    def _create_tags(tag, length):
        labels = ['B-' + tag]
        if length > 1:
            for _ in range(length - 1):
                labels.append('I-' + tag)
        return labels

    def load_data(self, training_data):
        sentences = []
        tags = []
        train_data = {}
        for ex in training_data.training_examples:
            intent = ex.get("intent")
            if intent not in train_data:
                sentences = []
            token = []
            for i in range(len(ex.get("tokens"))):
                token += [ex.get("tokens")[i].text]
            tags=['O'] * len(ex.text)
            if ex.get("entities") is not None:
                for entity in ex.get("entities"):
                    ent = entity.get('entity', None)
                    start = entity.get('start', None)
                    end = entity.get('end', None)
                    tags[start:end] = self._create_tags(ent, len(entity))
            sentences.append((token, tags))
            train_data[intent] = sentences

        train = [(t, l, i) for i in sorted(train_data) for t, l in train_data[i]]

        self.dataset = IntentDataset()
        self.dataset._load_data(train_set=train)

    def train(self, training_data, cfg=None, **kwargs):
        # type: (TrainingData, Optional[RasaNLUModelConfig], **Any) -> None
        """Train the embedding intent classifier on a data set."""

        self.load_data(training_data)

        # train tensorflow graph
        config_proto = self.get_config_proto(self.component_config)
        self.session = tf.Session(graph=self.graph, config=config_proto)

        # x, char, 意图, label
        train_x, train_char, train_i, train_y = self.dataset.train_set

        train_y = to_categorical(train_y, self.dataset.label_vocab_size)
        train_i = one_hot(train_i, len(self.dataset.intents_vocab))
        train_inputs = [train_x, train_char]
        train_outs = [train_i, train_y]

        self.model = MultiTaskIntentModel(use_cudnn=self.use_cudnn)
        self.model.build(self.dataset.word_len,
                    self.dataset.label_vocab_size,
                    self.dataset.intent_size,
                    self.dataset.word_vocab_size,
                    self.dataset.char_vocab_size,
                    word_emb_dims=self.token_emb_size,
                    tagger_lstm_dims=self.lstm_hidden_size,
                    dropout=self.tagger_dropout)

        # initialize word embedding if external model selected
        # 如果存在词向量model，就初始化词向量
        if self.embedding_model is not None:
            #print('Loading external word embedding')
            embedding_model, _ = load_word_embeddings(self.embedding_model)
            embedding_mat = get_embedding_matrix(embedding_model, self.dataset.word_vocab)
            self.model.load_embedding_weights(embedding_mat)

        # 即每次训练的时候会执行keras的callback函数
        conll_cb = ConllCallback(train_inputs, train_y, self.dataset.tags_vocab.vocab, batch_size=self.batch_size)

        # train model
        self.model.fit(x=train_inputs, y=train_outs,
                batch_size=self.batch_size, epochs=self.epochs,
                validation=(train_inputs, train_outs),
                callbacks=[conll_cb])
        #print('Training done')

    def vectorize(self, doc, vocab, char_vocab=None):
        words = np.asarray([vocab[w.lower()] if w.lower() in vocab else 1 for w in doc])\
            .reshape(1, -1)
        if char_vocab is not None:
            sentence_chars = []
            for w in doc:
                word_chars = []
                for c in w:
                    if c in char_vocab:
                        _cid = char_vocab[c]
                    else:
                        _cid = 1
                    word_chars.append(_cid)
                sentence_chars.append(word_chars)
            sentence_chars = np.expand_dims(pad_sentences(sentence_chars, self.model.word_length),
                                            axis=0)
            return [words, sentence_chars]
        else:
            return words

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        """Return the most likely intent and its similarity to the input."""

        intent = {"name": None, "confidence": 0.0}
        intent_ranking = []

        # get features (bag of words) for a message
        X = message.get("tokens")
        text_arr =[]
        intent_type = None
        for i in range(len(X)):
            text_arr += [X[i].text]

        # initialize word embedding if external model selected
        # 如果存在词向量model，就初始化词向量
        if self.embedding_model is not None:
            print('Loading external word embedding')
            embedding_model, _ = load_word_embeddings(self.embedding_model)
            embedding_mat = get_embedding_matrix(embedding_model, self.word_vocab)
            self.model.load_embedding_weights(embedding_mat)

        doc_vec = self.vectorize(text_arr, self.word_vocab, self.char_vocab)
        intents, tags = self.model.predict(doc_vec, batch_size=1)

        intent_id = int(intents.argmax(1).flatten())
        intent_type = self.intent_vocab.get(intent_id, None)
        #print('Detected intent type: {}'.format(intent_type))

        tags = tags.argmax(2).flatten()
        tag_str = [self.tags_vocab.get(n, None) for n in tags]
        #for t, n in zip(text_arr, tag_str):
        #    print('{}\t{}\t'.format(t, n))

        spans = []
        available_tags = set()

        for s, e, tag in bio_to_spans(text_arr, tag_str):
            spans.append({
                'start': s,
                'end': e,
                'type': tag
            })
            available_tags.add(tag)

        intent = {"name": intent_type, "confidence": float(intents[0][intent_id])}
        ranking = intents[0][:INTENT_RANKING_LENGTH]

        intent_ranking = [{"name": self.intent_vocab.get(intent_idx, None), "confidence": float(score)}
                        for intent_idx, score in enumerate(ranking)]

        intent_ranking = sorted(intent_ranking, key=lambda s: s['confidence'], reverse=True)

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)
        message.set("entities", message.get("entities", []) + list(available_tags), add_to_output=True)

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory.
        Return the metadata necessary to load the model again."""
        if self.session is None:
            return {"classifier_file": None}

        self.model_path = os.path.join(model_dir, self.name + ".h5")
        self.model_info_path = os.path.join(model_dir, self.name + ".dat")
        try:
            os.makedirs(os.path.dirname(self.model_path))
            os.makedirs(os.path.dirname(self.model_info_path))
        except OSError as e:
            # be happy if someone already created the path
            import errno
            if e.errno != errno.EEXIST:
                raise
        print('Saving model')

        self.model.save(self.model_path)
        with open(self.model_info_path, 'wb') as fp:
            info = {
                'type': 'mtl',
                'tags_vocab': self.dataset.tags_vocab.vocab,
                'word_vocab': self.dataset.word_vocab.vocab,
                'char_vocab': self.dataset.char_vocab.vocab,
                'intent_vocab': self.dataset.intents_vocab.vocab,
            }
            pickle.dump(info, fp)
        return {"classifier_file": self.name + ".h5","classifier_info_file": self.name + ".dat"}

    @staticmethod
    def get_config_proto(component_config):
        # 配置configProto
        config = tf.ConfigProto(
            device_count={
                'CPU': component_config['config_proto']['device_count']
            },
            inter_op_parallelism_threads=component_config['config_proto']['inter_op_parallelism_threads'],
            intra_op_parallelism_threads=component_config['config_proto']['intra_op_parallelism_threads'],
            gpu_options={
                'allow_growth': component_config['config_proto']['allow_growth']
            }
        )
        return config

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        # type: (...) -> EmbeddingIntentClassifier

        meta = model_metadata.for_component(cls.name)
        config_proto = cls.get_config_proto(meta)

        if model_dir and meta.get("classifier_file"):
            file_name = meta.get("classifier_file")
            model_path = os.path.join(model_dir, file_name)
            file_info_name = meta.get("classifier_info_file")
            model_info_path = os.path.join(model_dir, file_info_name)

            with open(model_info_path, 'rb') as fp:
                model_info = pickle.load(fp)
            cls.word_vocab = model_info['word_vocab']
            cls.tags_vocab = {v: k for k, v in model_info['tags_vocab'].items()}
            cls.char_vocab = model_info['char_vocab']
            cls.intent_vocab = {v: k for k, v in model_info['intent_vocab'].items()}
            graph = tf.Graph()
            with graph.as_default():
                sess = tf.Session(config=config_proto)
            cls.model = MultiTaskIntentModel()
            cls.model.load(model_path)

            return MultiTaskIntent(
                    component_config=meta,
                    session=sess,
                    graph=graph
            )

        else:
            logger.warning("Failed to load nlu model. Maybe path {} "
                           "doesn't exist"
                           "".format(os.path.abspath(model_dir)))
            return MultiTaskIntent(component_config=meta)
