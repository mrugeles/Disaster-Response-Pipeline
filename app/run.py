import sys
import json
import plotly
import re
import pandas as pd
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

nltk.download('stopwords')

app = Flask(__name__)

def tokenize(text):
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
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]
    words = stemmed = [PorterStemmer().stem(w) for w in words]
    return text

# load model
model = joblib.load("../models/classifier.pkl")

# Dataset categories
columns = ['related', 'request', 'offer',
   'aid_related', 'medical_help', 'medical_products',
   'search_and_rescue', 'security', 'military', 'child_alone', 'water',
   'food', 'shelter', 'clothing', 'money', 'missing_people',
   'refugees', 'death', 'other_aid', 'infrastructure_related',
   'transport', 'buildings', 'electricity', 'tools', 'hospitals',
   'shops', 'aid_centers', 'other_infrastructure', 'weather_related',
   'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather',
   'direct_report', 'genre_direct', 'genre_news', 'genre_social']

# CSS class icons for categories
category_icons = {
   'related':'exclamation-triangle', 'request':'hand-point-up', 'offer':'hand-holding-heart',
   'aid_related':'first-aid', 'medical_help':'briefcase-medical', 'medical_products':'prescription-bottle-alt',
   'search_and_rescue':'helicopter', 'security':'shield-alt', 'military':'exclamation-circle', 'child_alone':'child',
   'water':'tint', 'food':'utensils', 'shelter':'home', 'clothing':'tshirt', 'money':'hand-holding-usd',
   'missing_people':'user-minus', 'refugees':'user-friends', 'death':'user-times', 'other_aid':'hands-helping',
   'infrastructure_related':'house-damage', 'transport':'bus-alt', 'buildings':'building', 'electricity':'bolt',
   'tools':'tools', 'hospitals':'hospital', 'shops':'shopping-basket', 'aid_centers':'clinic-medical',
   'other_infrastructure':'city', 'weather_related':'cloud-sun-rain',
   'floods':'water', 'storm':'cloud-showers-heavy', 'fire':'burn', 'earthquake':'house-damage',
   'cold':'icicles', 'other_weather':'cloud-sun',
   'direct_report':'table', 'genre_direct':'mobile', 'genre_news':'newspaper', 'genre_social':'twitter'
}

# CSS class for category colors
category_colors = {
   'related':'primary', 'request':'info', 'offer':'success',
   'aid_related':'danger', 'medical_help':'warning', 'medical_products':'info',
   'search_and_rescue':'primary', 'security':'info', 'military':'success', 'child_alone':'danger',
   'water':'warning', 'food':'info', 'shelter':'primary', 'clothing':'info', 'money':'success',
   'missing_people':'danger', 'refugees':'warning', 'death':'info', 'other_aid':'primary',
   'infrastructure_related':'info', 'transport':'success', 'buildings':'danger', 'electricity':'warning',
   'tools':'info', 'hospitals':'primary', 'shops':'info', 'aid_centers':'success',
   'other_infrastructure':'danger', 'weather_related':'warning',
   'floods':'info', 'storm':'primary', 'fire':'info', 'earthquake':'success',
   'cold':'danger', 'other_weather':'warning',
   'direct_report':'info', 'genre_direct':'primary', 'genre_news':'info', 'genre_social':'success'
}

# Set random colors for category plots
import random as random
backgroundColors = []
borderColors = []
for category in columns:
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    backgroundColors.append( "rgba({}, {}, {}, 0.2)".format(r, g, b))
    borderColors.append( "rgb({}, {}, {})".format(r, g, b))

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Results are shown in a table and a plot to show the predicted categories.
    """
    query = request.args.get('query', '')
    classification_results = None
    categories = None
    categories_totals = None

    if(query != ''):
        # use model to predict classification for query
        classification_labels = model.predict([query])[0]
        classification_results = dict(zip(columns, classification_labels))

        columns_df = ['message']
        columns_df.extend(columns)

        messages_df = pd.DataFrame(columns = columns_df)
        messages_df = messages_df.append(classification_results, ignore_index = True)

        categories, categories_totals = get_plot_params(messages_df, True)

        store_message(query, messages_df)
    return render_template(
        'index.html',
        query=query,
        classification_result=classification_results,
        categories = categories,
        categories_totals = categories_totals,
        background_colors = backgroundColors,
        border_colors = borderColors
    )

def store_message(message, messages_df):
    """Every query is stored in a table with the predicted categories. This can eventually allow users to
    manually make correction to classifications that can be used to expand the training set.

    Parameters
    ----------
    message: string
        Message to classify.

    messages_df: DataFrame
        Dataframe with the predicted categories for the message.
    """
    engine = create_engine('sqlite:///ReceivedMessages.db')
    exists = 0
    try:
        messages = pd.read_sql_table('messages', engine)
        exists = messages.loc[messages['message'] == message].shape[0]
    except Exception as inst:
        print("Unexpected error:", inst)
        pass
    if(exists == 0):
        messages_df.to_sql('messages', engine, index=False, if_exists='append')

def get_plot_params(messages, filter = False):
    """ Returns parameter values for build a chartjs plot

        Parameters
        ----------
        messages: DataFrame
            DataFrame with messages and their predicted categories.

        filter: bool
            If True, only returns categories with value 1.

        Returns
        -------
        categories: array
            Ids for predicted categories.
        categories_totals: array
            Total of records per categories.
    """
    categories_df = messages.drop(['message'], axis = 1)
    totals = categories_df.sum()

    categories_totals_df = pd.DataFrame({'category':totals.index, 'total':totals.values})
    if(filter == True):
        categories_totals_df = categories_totals_df.loc[categories_totals_df['total'] > 0]
    categories = list(categories_totals_df['category'].values)
    categories_totals = list(categories_totals_df['total'].values)
    return categories, categories_totals

def get_top_categories(messages, top):
    """ Return the most required categories

    Parameters
    ----------
    messages: DataFrame
        DataFrame with messages and their predicted categories.

    top: int
        Number of categories to retrieve.

    Returns
    -------
    categories_totals_df: DataFrame
        Dataframe with the most required categories.
    """
    categories_df = messages.drop(['message'], axis = 1)
    totals = categories_df.sum()

    categories_totals_df = pd.DataFrame({'category':totals.index, 'total':totals.values})
    return categories_totals_df.sort_values(by=['total'], ascending = False).head(n = top)

@app.route('/dashboard')
def dashboard():
    """
    This page provides basic plotting information from the messages received and their classifications.
    """
    engine = create_engine('sqlite:///ReceivedMessages.db')
    categories_top_df = pd.DataFrame()
    categories_totals_df = pd.DataFrame()
    categories = None
    categories_totals = None

    try:
        received_messages = pd.read_sql_table('messages', engine)
        categories, categories_totals = get_plot_params(received_messages)
        categories_top_df = get_top_categories(received_messages, 4)

    except:
        print("Unexpected error:", sys.exc_info()[0])
        pass
    return render_template(
        'dashboard.html',
        categories_top_df = categories_top_df,
        category_icons = category_icons,
        category_colors = category_colors,
        categories = categories,
        categories_totals = categories_totals,
        background_colors = backgroundColors,
        border_colors = borderColors
    )

@app.route('/dataset')
def dataset():
    return render_template(
        'dataset.html'
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
