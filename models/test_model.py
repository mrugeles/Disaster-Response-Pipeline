import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import nltk
import re

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.externals import joblib
from nlp_utils import NLPUtils
nlpUtils = NLPUtils()
#features_corpus = pd.read_csv('data/features_corpus.csv')


columns = ['related', 'request', 'offer',
   'aid_related', 'medical_help', 'medical_products',
   'search_and_rescue', 'security', 'military', 'child_alone', 'water',
   'food', 'shelter', 'clothing', 'money', 'missing_people',
   'refugees', 'death', 'other_aid', 'infrastructure_related',
   'transport', 'buildings', 'electricity', 'tools', 'hospitals',
   'shops', 'aid_centers', 'other_infrastructure', 'weather_related',
   'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather',
   'direct_report', 'genre_direct', 'genre_news', 'genre_social']


def main():

   


    model = joblib.load('models/classifier.pkl')
    text = 'After the earthquake i cannot call any body. even when i ihave 10 signal'
    X = [text]
    query = pd.DataFrame(X, columns=['document'])
    query = nlpUtils.vectorize_data(query['document'], 'count_vectorizer.p')
    y_pred = model.predict(query)[0]
    classification_results = dict(zip(columns, y_pred))
    classification_results =  {key:value for (key,value) in classification_results.items() if value == 1}
    print(f'query: {text}')
    print(y_pred)
    print(classification_results)

if __name__ == '__main__':
    main()
