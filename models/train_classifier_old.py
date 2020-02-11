import sys
import os
from metaflow import FlowSpec, Parameter, step, resources
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from time import time

import nltk
nltk.download('words')

import pandas as pd
import numpy as np

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


class ModelFlow(FlowSpec):

    database_filepath = 'data/DisasterResponse.db'
    model_filepath = 'models/classifier.pkl'

    def __init__(self):
        self.modelUtils = ModelUtils()
        self.dataUtils = DataUtils()
        self.nlpUtils = NLPUtils()
        self.X = None
        self.Y = None
        self.category_names = None

    def start(self):
        self.load_data()
        self.vectorize()
        self.clean_vector()
        self.normalize()
        self.build_model()
        self.save_model()


    def load_data(self):
        self.X, self.Y, self.category_names = self.dataUtils.load_db_data(self.database_filepath)


    def vectorize(self):
        start = time()
        self.X = self.nlpUtils.create_vector_model(self.X)
        end = time()
        print(f'vectorize time: {end - start}')

    def clean_vector(self):
        start = time()
        self.X = self.nlpUtils.clean_count_vector(self.X)
        end = time()
        print(f'clean_vector time: {end - start}')

    def normalize(self):
        start = time()
        self.X = self.nlpUtils.normalize_count_vector(self.X)
        end = time()
        print(f'normalize time: {end - start}')
        
    def build_model(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2)
        self.model = MultiOutputClassifier(RandomForestClassifier())
        self.model.fit(self.X_train, self.Y_train)
        self.modelUtils.evaluate_model(self.model, self.X_test, self.Y_test, self.category_names)

    def save_model(self):
        self.modelUtils.save_model(self.model, self.model_filepath)



if __name__ == '__main__':
    modelFlow = ModelFlow()
    modelFlow.start()
