
import re

import pandas as pd
import numpy as np
import pickle

from tqdm import tqdm
from time import time
import logging

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer

from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from scipy.sparse import csr_matrix

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class NLPUtils():

    english_corpus = set(words.words())

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
        text = text.lower()
        text = re.sub(r"[^a-z]", " ", text)
        #text = TextBlob(text).correct().string
        word_list = word_tokenize(text)
        word_list = [w for w in word_list if w not in stopwords.words("english")]
        word_list = [WordNetLemmatizer().lemmatize(w, pos='v') for w in word_list]
        #word_list = list(set(word_list).intersection(self.english_corpus))
        return word_list

        


    def create_vector_model(self, features, features_corpus_path, pickle_path):
        start = time()
        english_corpus = set(words.words())

        features_corpus = pd.read_csv(features_corpus_path)
        count_vect = CountVectorizer(tokenizer=self.tokenize)
        count_vect = count_vect.fit(features_corpus['document'])

        vectorized = count_vect.transform(features)

        matrix = pd.DataFrame(vectorized.toarray(), columns=count_vect.get_feature_names())
        word_list = list(matrix.columns)
        word_list = [TextBlob(word).correct().string for word in tqdm(word_list)]
        matrix.columns = word_list
        word_list = list(set(word_list).intersection(english_corpus))
        word_list.sort()
        matrix = matrix[word_list]
        matrix = csr_matrix(matrix.values)

        vector = TfidfTransformer().fit(matrix)
        pickle.dump(vector, open(pickle_path, "wb"))
        
        end = time()

        print(f'Vectorizing time: {end - start}')

        return vector

    def vectorize_data(self, data, pickle_path):
        vectorizer = pickle.load(open( pickle_path, "rb" ))
        print(f'type: {type(vectorizer)}')
        return vectorizer.transform(data)

