# DataScience
# Build a Machine Learning Pipeline to categorize emergency messages
A Udacity Data Scientist Nanodegree Project

![Alt text](./img/Screenshot_Disasters.png?raw=true "Message Classifier")

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)
6. [References](#references)

## Installation <a name="installation"></a>
1. Run the following commands from the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command from the app's directory to run the web app.
    `python run.py`

3. Find the workspace environmental variables with env | grep WORK, and you can open a new browser window and go to the address: http://WORKSPACESPACEID-3001.WORKSPACEDOMAIN replacing WORKSPACEID and WORKSPACEDOMAIN with your values.

## Project Motivation<a name="motivation"></a>
In this project I have built a machine learning pipeline to categorize emergency messages based on the needs communicated by the sender.
I have analyzed disaster data from Figure Eight to build a model for an API that classifies disaster messages.
From a data set containing real messages that were sent during disaster events, I have created a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

This project also includes a web app where an emergency worker can enter a message and get the classification results in several categories. The web app will also displays some nice visualizations of the data. So this project covers software skills, including the ability to create basic data pipelines and write clean, organized code.

Below are a few screenshots of the web app.


## File Descriptions <a name="files"></a>
Here's the file structure of the project:
![Alt text](./img/tree_Disaster.png?raw=true "Structure Project")
The main components for this project are:
1. ETL Pipeline

In the Python scrip: process_data.py, I've developed a data cleaning pipeline that:

    * Loads the source datasets
    * Merges them with an inner join through the id field
    * Tranforms them (convert categories into 0/1 numbers, drop some innecessary fields, concat fields,etc..)
    * Cleans all data (remove duplicates)
    * Saves it as a SQLite database with the "to_sql" method from the SQLAlchemy library


2. ML Pipeline

The Python script: train_classifier.py, contains a machine learning pipeline that:

    * Loads data from the database file previously prepared with the previous step
    * Split the data into a training set and a test set
    * Creates a ML pipeline that uses NLTK, scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification)
    * Finally that final model is exported to a pickle file


3. Flask Web App

    There is a flask web app that implements the developed ML model on a web page: you can enter a message and it predicts the category of the message.
    Also it shows some nice visualization using Plotly.

    ![Alt text](./img/Graph3.png?raw=true "Nice Visualizations")

## Results<a name="results"></a>
The main findings are explained on the Notebooks files that are included in the corresponding folder on this github repo.
I have included a minimal Exploratory Data Analysis:
![Alt text](./img/eda.png?raw=true "Overview")
![Alt text](./img/eda2.png?raw=true "Pearson")


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Figure Eight for collecting data and to Udacity because there are some pieces of code taken out from the DataScience Udacity Nanodegree classrooms.
Otherwise, feel free to use the code here as you would like!

## References <a name="references"></a>
 [Figure Eight](https://www.figure-eight.com/) <br>
 [Data Science Udacity Nanodegrees](https://www.udacity.com/school-of-data-science) <br>
