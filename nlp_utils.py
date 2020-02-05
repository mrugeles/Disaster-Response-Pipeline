
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

    def create_countvectorizer(self, features_corpus_path, pickle_path):
        features_corpus = pd.read_csv(features_corpus_path)
        count_vect = CountVectorizer(tokenizer=self.tokenize)
        count_vect = count_vect.fit(features_corpus['document'])
        pickle.dump(count_vect, open(pickle_path, "wb"))


    def vectorize(self, features, pickle_path):
        start = time()
        english_corpus = set(words.words())

        count_vect = pickle.load(pickle_path)
        vectorized = count_vect.transform(features)

        matrix = pd.DataFrame(vectorized.toarray(), columns=count_vect.get_feature_names())
        word_list = list(matrix.columns)
        word_list = [TextBlob(word).correct().string for word in tqdm(word_list)]
        matrix.columns = word_list
        word_list = list(set(word_list).intersection(english_corpus))
        word_list.sort()
        matrix = matrix[word_list]

        vector = TfidfTransformer().fit_transform(csr_matrix(matrix.values))
        end = time()

        print(f'Vectorizing time: {end - start}')

        return vector
