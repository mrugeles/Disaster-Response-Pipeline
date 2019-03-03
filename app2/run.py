import sys
import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

columns = ['related', 'request', 'offer',
   'aid_related', 'medical_help', 'medical_products',
   'search_and_rescue', 'security', 'military', 'child_alone', 'water',
   'food', 'shelter', 'clothing', 'money', 'missing_people',
   'refugees', 'death', 'other_aid', 'infrastructure_related',
   'transport', 'buildings', 'electricity', 'tools', 'hospitals',
   'shops', 'aid_centers', 'other_infrastructure', 'weather_related',
   'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather',
   'direct_report', 'genre_direct', 'genre_news', 'genre_social']
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
    query = request.args.get('query', '')
    classification_results = None
    if(query != ''):
        # use model to predict classification for query
        classification_labels = model.predict([query])
        classification_results = dict(zip(columns, classification_labels[0]))
        store_message(query, classification_results)
    return render_template(
        'index.html',
        query=query,
        classification_result=classification_results
    )

def store_message(message, classification_result):
    engine = create_engine('sqlite:///ReceivedMessages.db')
    exists = 0
    try:
        messages = pd.read_sql_table('messages', engine)
        exists = messages.loc[messages['message'] == message].shape[0]
    except Exception as inst:
        print("Unexpected error:", inst)
        pass
    if(exists == 0):
        columns_df = ['message']
        columns_df.extend(columns)

        classification_result['message'] = message
        messages_df = pd.DataFrame(columns = columns_df)

        messages_df = messages_df.append(classification_result, ignore_index = True)

        messages_df.to_sql('messages', engine, index=False, if_exists='append')


@app.route('/dashboard')
def dashboard():
    engine = create_engine('sqlite:///ReceivedMessages.db')
    categories_top_df = pd.DataFrame()
    categories_totals_df = pd.DataFrame()
    categories = None
    categories_totals = None

    try:
        received_messages = pd.read_sql_table('messages', engine)

        categories_df = received_messages.drop(['message'], axis = 1)
        print(categories_df.head())
        totals = categories_df.sum()

        categories_totals_df = pd.DataFrame({'category':totals.index, 'total':totals.values})
        categories = list(categories_totals_df['category'].values)
        categories_totals = list(categories_totals_df['total'].values)
        categories_top_df = categories_totals_df.sort_values(by=['total'], ascending = False).head(n = 4)

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
