"""
This script:
1. Loads data from a SQLite database
2. Splits the dataset into training and test sets
3. Builds a text processing & a machine learning pipeline
4. Trains and tunes a model (optimizing the hyperparameters) using the GridSearchCV method
5. Outputs results on the test set
6. Exports the final model as a pickle file
"""
# import basic libraries
import sys
import os
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
# import nltk and text processing (like regular expresion) libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download(['punkt', 'wordnet','stopwords'])
import re

# import libraries for transformation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# import machine learning libraries
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, fbeta_score, make_scorer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def load_data(db_file):
    """
    Load data function

    This method receives a database file on a path and it loads data
    from that database file into a pandas datafile.
    It also splits the data into X and y (X: features to work and y: labels to predict)
    It returns two sets of data: X and y

    Args:
        db_file (str): Filepath where database is stored.

    Returns:
        X (DataFrame): Feature columns
        y (DataFrame): Label columns

    """
    # load data from database
    # db_file = "./CleanDisasterResponse.db"
    # create the connection to the DB
    engine = create_engine('sqlite:///{}'.format(db_file))
    table_name = os.path.basename(db_file).replace(".db","")
    # load the info from the sql table into a pandas file
    df = pd.read_sql_table(table_name,engine)
    # We separate the features from the variables we are going to predict
    X = df ['message']
    y = df.drop(columns = ['id', 'message', 'original', 'genre'])

    return X, y


def tokenize(text):
    """
    Tokenize function

    Args:
        text: This method receives a text and it tokenizes it

    Returns:
        tokens: a set of tokens

    """
    # initialize language and WordNetLemmatizer
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

def build_model():
    """
    Build model procedure:

    This procedure builds a Machine Learning model based on a sklearn pipeline
    and using tfidf, random forest, and gridsearch

    Args: no args

    Returns:
    model (Scikit Pipeline Object and RandomForest Classifier) : ML Model
    """

    base = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    parameters = {'clf__estimator__n_estimators': [50, 100],
                  'clf__estimator__min_samples_split': [2, 3, 4],
                  'clf__estimator__criterion': ['entropy', 'gini']
                 }

    model = GridSearchCV(base, param_grid=parameters, n_jobs=-1, cv=2, verbose=10)

    return (model)


def display_results(y, y_test, y_pred):
    """
    Display results procedure:

    This procedure displays some metrics of the Machine Learning model

    Args: y, y_test and y_pred

    Returns:
        nothing, it displays some metrics like the Classification report and accuracy
    """

    category_names = list(y.columns)

    for i in range(len(category_names)):
        print("Output Category:", category_names[i],"\n", classification_report(y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(y_test.iloc[:, i].values, y_pred[:,i])))

def evaluate_model(model, y, x_test, y_test):
    """
    Evaluate model procedure:

    This procedure runs the model on a dataset and it displays the model's accuracy

    Args: model, X set, y set

    Returns:
        nothing, it runs the model and it displays accuracy metrics
    """

    # predict on test set data
    print ("Starting the prediction ...\n")
    y_pred = model.predict(x_test)
    print ("Finishing the prediction ...\n")
    # display metrics
    print ("Starting displaying accuracy results ...\n")
    display_results(y, y_test, y_pred)


def save_model(model, model_filepath):
    """
    Save the model as a pickle file:

    This procedure saves the model as a pickle file

    Args: model, X set, y set

    Returns:
        nothing, it runs the model and it displays accuracy metrics
    """
    try:
        pickle.dump(model, open(model_filepath, 'wb'))
    except:
        print("Error saving the model as a {} pickle file".format(model_filepath))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        X,y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model,y, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py data/DisasterResponse.db models/classifier.pkl')


if __name__ == '__main__':
    main()
