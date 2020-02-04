
import re

import pandas as pd
import numpy as np
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
        text = re.sub(r"[^a-z]", " ", text)
        word_list = word_tokenize(text)
        word_list = [w for w in word_list if w not in stopwords.words("english")]
        word_list = [WordNetLemmatizer().lemmatize(w, pos='v') for w in word_list]
        #word_list = set(word_list).intersection(self.english_corpus)
        #words = [PorterStemmer().stem(w) for w in words]
        return list(word_list)

    def vectorize(self, features):
        english_corpus = set(words.words())

        count_vect = CountVectorizer(tokenizer=self.tokenize)
        vectorized = count_vect.fit_transform(features)
        matrix = pd.DataFrame(vectorized.toarray(), columns=count_vect.get_feature_names())
        word_list = list(matrix.columns)
        word_list = list(set(word_list).intersection(english_corpus))
        word_list.sort()
        matrix = matrix[word_list]

        return TfidfTransformer().fit_transform(csr_matrix(matrix.values))
