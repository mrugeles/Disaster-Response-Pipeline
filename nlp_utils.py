import re

import pandas as pd
import numpy as np
import pickle

from time import time

import logging

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer

from textblob import TextBlob
from textblob import Word

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from scipy.sparse import csr_matrix
from wordsegment import load, segment
load()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class NLPUtils():

    def __init__(self):
        print('Init NLPUtils')
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.pattern = re.compile(r'(\#\w+)')
        


    def tokenize(self, text):
        """Text tokenization

        Parameters
        ----------
        text: string
            Text to tokenize

        Returns
        -------
        text: Tokenized text.
        """
        stopwords_list = stopwords.words("english")
        stopwords_list += ['null', 'nan']

        text = text.lower()

        hashtags = [hashtag.replace('#', '') for hashtag in re.findall(self.pattern, text)]
        stopwords_list += hashtags
        hashtags = ' '.join(hashtags)
        hashtags = segment(hashtags)

        text = re.sub(r"[^a-z]", " ", text)
        word_list = word_tokenize(text)
        word_list += hashtags
        word_list = [w for w in word_list if w not in stopwords_list]
        word_list = [WordNetLemmatizer().lemmatize(w, pos='v') for w in word_list]
        return word_list

        

    def spellcheck(self, word, threshold):
        value = Word(word).spellcheck()[0]
        if (value[1] < threshold):
            return f'-1'
        return value[0]


    def correct(self, text):
        return TextBlob(text).correct().string


    def process_words(self, columns):
        word_list = [(word, self.spellcheck(word, 0.7)) for word in columns]
        word_list = dict(word_list)
        return word_list


    def create_vector_model(self, features):

        count_vect = CountVectorizer(tokenizer=self.tokenize)
        count_vect = count_vect.fit(features)

        vectorized = count_vect.transform(features)
        matrix = pd.DataFrame(vectorized.toarray(), columns=count_vect.get_feature_names())

        return matrix


    def drop_duplicated(self, matrix):
        processed_columns = []    

        for column in matrix.columns[matrix.columns.duplicated()]:
            if(column not in processed_columns):
                column_indexes = np.where(matrix.columns.values == column)[0]
                
                for idx in column_indexes[1:]:
                    matrix.iloc[:, column_indexes[0]] = matrix.iloc[:, column_indexes[0]].combine(matrix.iloc[:, idx], max)
                processed_columns += [column]
        matrix = matrix.iloc[:,~matrix.columns.duplicated()]
        return matrix


    def clean_count_vector(self, matrix):
        start = time()
        columns_df = pd.DataFrame(list(matrix.columns), columns = ['feature'])
        columns_df['feature_spellcheck'] = columns_df['feature'].apply(lambda word: self.spellcheck(word, 0.7))

        start = time()
        drop_columns = columns_df.loc[
            (columns_df['feature_spellcheck'] == "-1") |
            (columns_df['feature_spellcheck'] == "null") |
            (columns_df['feature_spellcheck'] == "nan")
        ]['feature'].values
        matrix = matrix.drop(drop_columns, axis = 1)

        start = time()
        renamed_columns = dict(columns_df.loc[columns_df['feature_spellcheck'] != "-1"].to_dict('split')['data'])
        matrix = matrix.rename(columns = renamed_columns)

        matrix = self.drop_duplicated(matrix)

        matrix = matrix.reindex(sorted(matrix.columns), axis=1)
        
        model_features = pd.DataFrame(matrix.columns, columns = ['feature'])
        model_features.to_csv('models/model_features.csv', index = False)
        
        return csr_matrix(matrix.values)

    def normalize_count_vector(self, count_matrix):
        vectorizer = TfidfTransformer().fit(count_matrix)
        pickle.dump(vectorizer, open('models/tfidf-vector.p', "wb"))
        matrix = vectorizer.transform(count_matrix)
        return matrix


    def get_matrix(self, data):
        count_vect = CountVectorizer(tokenizer=self.tokenize)
        vectorized = count_vect.fit_transform(data)
        return pd.DataFrame(vectorized.toarray(), columns=count_vect.get_feature_names())

    def vectorize_query(self, query):
        query = TextBlob(query).correct().string
        matrix_query = self.get_matrix([query])
        model_features = pd.read_csv('models/model_features.csv')
        
        model_features = list(model_features['feature'].values)

        add_features = list(set(model_features).difference(set(matrix_query.columns)))
        remove_features = list(set(matrix_query.columns).difference(set(model_features)))

        n_features = len(add_features)
        n_rows = matrix_query.shape[0]

        matrix_query = matrix_query.drop(remove_features, axis = 1)
        matrix_query[add_features] = pd.DataFrame(np.zeros((n_rows, n_features), dtype = int), columns = add_features)
        
        matrix_query = matrix_query.reindex(sorted(matrix_query.columns), axis=1)
        
        matrix_query = csr_matrix(matrix_query.values)
        vectorizer = pickle.load( open( 'models/tfidf-vector.p', "rb" ) )
        
        return vectorizer.transform(matrix_query)
