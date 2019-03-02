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
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_direct_counts = df.loc[df['genre_direct'] == 1].shape[0]
    genre_news_counts = df.loc[df['genre_news'] == 1].shape[0]
    genre_social_counts = df.loc[df['genre_social'] == 1].shape[0]

    genre_counts = [genre_direct_counts, genre_news_counts, genre_social_counts]
    genre_names = ['direct', 'news', 'social']
    #genre_counts = df.groupby('genre').count()['message']
    #genre_names = list(genre_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query')
    # use model to predict classification for query
    print("query:")
    print(query)
    classification_labels = model.predict([query])
    print(classification_labels[0])
    print( model.predict([query]))
    classification_results = dict(zip(columns, classification_labels[0]))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
