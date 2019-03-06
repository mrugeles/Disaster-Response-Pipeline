import sys
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

columns = ['related', 'request', 'offer',
   'aid_related', 'medical_help', 'medical_products',
   'search_and_rescue', 'security', 'military', 'child_alone', 'water',
   'food', 'shelter', 'clothing', 'money', 'missing_people',
   'refugees', 'death', 'other_aid', 'infrastructure_related',
   'transport', 'buildings', 'electricity', 'tools', 'hospitals',
   'shops', 'aid_centers', 'other_infrastructure', 'weather_related',
   'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather',
   'direct_report', 'genre_direct', 'genre_news', 'genre_social']

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]
    words = stemmed = [PorterStemmer().stem(w) for w in words]
    return text

def main():
    model = joblib.load('models/classifier.pkl')
    X = ['we need medicines, can anyone help us with that?']
    y_pred = model.predict(X)[0]
    print(y_pred)

if __name__ == '__main__':
    main()
