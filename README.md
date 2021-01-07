# Disaster Response Pipeline Project
In this project I have built a machine learning pipeline to categorize emergency messages based on the needs communicated by the sender.
I have applied my skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.
From a data set containing real messages that were sent during disaster events, I have created a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

This project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. So this project cover software skills, including the ability to create basic data pipelines and write clean, organized code.

Below are a few screenshots of the web app.

There are three main components for this project:
1. ETL Pipeline

In a Python script: process_data.py, there is a data cleaning pipeline that:

    Loads the messages and categories datasets
    Merges the two datasets
    Cleans the data
    Stores it in a SQLite database

2. ML Pipeline

In a Python script: train_classifier.py, there is a machine learning pipeline that:

    Loads data from the SQLite database
    Splits the dataset into training and test sets
    Builds a text processing and machine learning pipeline
    Trains and tunes a model using GridSearchCV
    Outputs results on the test set
    Exports the final model as a pickle file

3. Flask Web App

    There is a flask web app to show some nice visualization using Plotly.

Here's the file structure of the project:



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Find the workspace environmental variables with env | grep WORK, and you can open a new browser window and go to the address: http://WORKSPACESPACEID-3001.WORKSPACEDOMAIN replacing WORKSPACEID and WORKSPACEDOMAIN with your values.
