import json
import plotly
import pandas as pd

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
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    top_10_mes = df.iloc[:,4:].sum().sort_values(ascending=False)[0:10]
    top_10_mes_names = list(top_10_mes.index)

    bottom_10_mes = df.iloc[:,5:].sum().sort_values()[0:10]
    bottom_10_mes_names = list(bottom_10_mes.index)

    mes_categories = df.columns[4:-1]
    mes_categories_count = df[mes_categories].sum()

    distr_class_1 = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()/len(df)
    distr_class_1.sort_values(ascending = False)
    distr_class_0 = 1 - distr_class_1
    distr_class_names = list(distr_class_1.index)

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
        },
                {
            'data': [
                Bar(
                    x=distr_class_names,
                    y=distr_class_1,
                    name = 'Class = 1',
                    marker=dict(color='green')
                ),
                Bar(
                    x=distr_class_names,
                    y=distr_class_0,
                    name = 'Class = 0',
                    marker = dict(color = 'rgb(210, 220, 240)')
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories within classes',
                'yaxis': {
                    'title': "Distribution"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': "-45"
                },
                'barmode' : 'stack'
            }
        },

        {
            'data': [
                Bar(
                    x=top_10_mes_names,
                    y=top_10_mes,
                    marker=dict(color='crimson')
                )
            ],

            'layout': {
                'title': 'Top Ten more frequent Disaster Messages Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle':"-45"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=bottom_10_mes_names,
                    y=bottom_10_mes,
                    marker=dict(color='rgb(0, 134, 149)')
                )
            ],

            'layout': {
                'title': 'Ten less frequent Disaster Messages Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle':"-50"
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
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

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
