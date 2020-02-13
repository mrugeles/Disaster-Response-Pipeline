import sys
import os
from metaflow import FlowSpec, Parameter, step
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from time import time

import nltk
nltk.download('words')

import pandas as pd
import numpy as np
from tqdm import tqdm

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from data_utils import DataUtils
from model_utils import ModelUtils
from nlp_utils import NLPUtils

import warnings
warnings.filterwarnings('ignore')

tqdm.pandas(desc="feature_spellcheck")

class ModelFlow(FlowSpec):

    fraction = Parameter('fraction',
                    help='Dataset sample size',
                    type=float)

    version_name = Parameter('version_name',
                    help='A version name for this flow',
                    type=str)

    database_filepath = Parameter('database_filepath',
                    help='Filepath of the disaster messages database',
                    type=str)
    model_filepath = Parameter('model_filepath',
                help='Filepath of the pickle file to save the model',
                type=str)

    @step
    def start(self):
        self.modelUtils = ModelUtils()
        self.dataUtils = DataUtils()
        self.nlpUtils = NLPUtils()
        self.next(self.load_data)

    @step
    def load_data(self):
        self.X, self.Y, self.category_names = self.dataUtils.load_db_data(self.database_filepath, self.fraction)
        self.next(self.vectorize)


    @step
    def vectorize(self):
        self.X = self.nlpUtils.create_vector_model(self.X)
        self.next(self.clean_vector)

    @step
    def clean_vector(self):
        self.X = self.nlpUtils.clean_count_vector(self.X)
        self.next(self.normalize)

    @step
    def normalize(self):
        self.X = self.nlpUtils.normalize_count_vector(self.X)
        self.next(self.build_model)

    @step   
    def build_model(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2)
        self.model = MultiOutputClassifier(RandomForestClassifier())
        self.model.fit(self.X_train, self.Y_train)
        self.scores = self.modelUtils.evaluate_model(self.model, self.X_test, self.Y_test, self.category_names)
        self.next(self.save_model)

    @step
    def save_model(self):
        self.modelUtils.save_model(self.model, self.model_filepath)
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    ModelFlow()
