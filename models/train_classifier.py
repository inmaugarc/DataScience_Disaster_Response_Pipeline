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
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


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

    This procedure builds the Machine Learning model

    Args: no args

    Returns:
    model (Scikit Pipelin Object and RandomForest Classifier) : ML Model
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier (RandomForestClassifier()))
    ])
    return (pipeline)

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



def old_display_results(y_test, y_pred):
    """
    Display results procedure:

    This procedure displays some metrics of the Machine Learning model

    Args: y_test and y_pred

    Returns:
        nothing, it displays some metrics like the Confussion Matrix and accuracy
    """

    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)


def evaluate_model(model, X, y):
    """
    Evaluate model procedure:

    This procedure runs the model on a dataset and it displays the model's accuracy

    Args: model, X set, y set

    Returns:
        nothing, it runs the model and it displays accuracy metrics
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # train classifier
    pipeline = build_model()
    pipeline.fit(X_train, y_train)
    # predict on test set data
    y_pred = pipeline.predict(X_test)
    # display metrics
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
        #X, Y, category_names = load_data(database_filepath)
        X,Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
       # evaluate_model(model, X_test, Y_test, category_names)
        evaluate_model(model, X_test, Y_test)

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
