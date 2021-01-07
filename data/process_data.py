# Import the libraries we are going to use
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load messages_file and categories_file
    Args:
        messages_filepath: csv file with the disaster messages
        categories_filepath: csv file with the categories for disaster messages
    Returns: a file that is merge of the two input files
    """

    df_mes = pd.read_csv(messages_filepath)
    df_cat = pd.read_csv(categories_filepath)
    df = df_mes.merge(df_cat,how="inner", on=["id"])

    return df


def clean_data(df):
    """
    This script cleans data
    Args:
        pandas-dataframe file: it contains the data to clean
    Returns: a dataframe file with clean data
    """
    # Let's split the categories into separated columns
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=";",expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[[0]]

    # use this row to extract a list of new column names for categories.
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # replace categories column in df with new category columns
    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],join="inner",axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True);

    return df


def save_data(df, database_filename):
        """
        This script saves cleaned data in to a database file
        Args:
            pandas-dataframe file: it contains the already cleaned data
            database-filename: path to the database file where we will save data
        Returns: ?
        """
        # save the clean dataset into an sqlite database
        engine = create_engine('sqlite:///'+ database_filename)
        df.to_sql('DisasterResponse', engine, index=False)


def main():
    """
        This main funtion runs all data processing, that is, loads 3 files:
            - the messages file in a csv format
            - the categories file in a csv format
            - the db file
        Then it cleans and prepares data for the next phases of this project
        And it saves the cleaned data on a SQLite destination database
    Args:
        messages_file: file that contains the disaster messages
        categories_file: file that contains the category of disaster messages
        database_file: path to the database file where clean data will be saved

   Returns: nothing

        Script execution syntax:
            python process_data.py <path_to_messages_file.csv> <path_to_categories_file.csv> <path_to_SQLite_response_database.db>
        Sample script execution:
            python process_data.py disaster_messages.csv disaster_categories.csv disaster_response.db

    """

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
