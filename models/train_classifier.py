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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    print('sqlite:///'+database_filepath)
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('messages', engine)
    X = df[['message']].values.flatten()
    y = df.drop(['id', 'message', 'original'], axis = 1)
    category_names = list(y.columns.values)
    #y = y.values

    return X, y, category_names


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]
    words = stemmed = [PorterStemmer().stem(w) for w in words]
    return text


def build_model():
    forest = RandomForestClassifier()
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(forest))
    ])
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__max_features': ['auto', 'sqrt', 'log2'],
        'clf__estimator__max_depth' : [3, 5],
        'clf__estimator__criterion' :['gini', 'entropy']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(data = y_pred, columns = category_names)
    for category in category_names:
        print("Scoring %s"%(category))
        print(classification_report(Y_test[category].values, y_pred[category].astype(int)))


def save_model(model, model_filepath):
    import pickle
    # save the classifier
    pickle.dump(model, open(model_filepath, 'wb'))
    return True


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')

        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
