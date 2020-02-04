import sys
import os
from metaflow import FlowSpec, Parameter, step
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import nltk

import numpy as np

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from data_utils import DataUtils
from model_utils import ModelUtils
from nlp_utils import NLPUtils


class ModelFlow(FlowSpec):

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
        self.X, self.Y, self.category_names = self.dataUtils.load_db_data(self.database_filepath)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2)
        self.next(self.pre_process)

    @step
    def pre_process(self):
        self.X_train = self.nlpUtils.vectorize(self.X_train)
        self.next(self.build_model)


    @step
    def build_model(self):
        self.model = self.modelUtils.build_model()
        self.model.fit(self.X_train, self.Y_train)
        self.modelUtils.evaluate_model(self.model, self.X_test, self.Y_test, self.category_names)
        self.next(self.save_model)

    @step
    def save_model(self):
        self.modelUtils.save_model(self.model, self.model_filepath)
        self.next(self.end)

    @step
    def end(self):
        pass

    '''def main():
        if len(sys.argv) == 3:
            database_filepath, model_filepath = sys.argv[1:]
            print('Loading data...\n    DATABASE: {}'.format(database_filepath))
            X, Y, category_names = dataUtils.load_db_data(database_filepath)

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

            print('Building model...')
            model = modelUtils.build_model()

            print('Training model...')

            model.fit(X_train, Y_train)

            print('Evaluating model...')
            modelUtils.evaluate_model(model, X_test, Y_test, category_names)

            print('Saving model...\n    MODEL: {}'.format(model_filepath))
            modelUtils.save_model(model, model_filepath)

            print('Trained model saved!')

        else:
            print('Please provide the filepath of the disaster messages database '\
                'as the first argument and the filepath of the pickle file to '\
                'save the model to as the second argument. \n\nExample: python '\
                'train_classifier.py ../data/DisasterResponse.db classifier.pkl')'''


if __name__ == '__main__':
    ModelFlow()
