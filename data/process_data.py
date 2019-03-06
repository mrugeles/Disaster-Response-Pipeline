import sys
import pandas as pd

def get_encodings(filepath):
    """ Prints encodings related with a given file

    Parameters
    ----------
    filepath: string
        Path to the file to analyse.
    """
    from encodings.aliases import aliases

    alias_values = set(aliases.values())

    for encoding in set(aliases.values()):
        try:
            df=pd.read_csv(filepath, encoding=encoding)
            print('successful', encoding)
        except:
            pass

def load_data(messages_filepath, categories_filepath):
    """Creates the dataframe for the pipeline

    Parameters
    ----------
    messages_filepath: string
        Path to the messages's csv file.

    categories_filepath: string
        Path to the categories's csv file.

    Returns
    -------
    df: DataFrame
        Dataframe with messages and it's categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_on='id', right_on='id')

    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = list(row.apply(lambda category_value: category_value.split('-')[0]))
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].apply(lambda text: text.split('-')[1])
        categories[column] = categories[column].astype(int)
    df = df.drop(['categories'], axis = 1)
    df = pd.concat([df, categories], axis = 1)

    genres = pd.get_dummies(df[['genre']])
    df = df.drop(['genre'], axis = 1)
    df = pd.concat([df, genres], axis = 1)
    return df

def clean_data(df):
    """This method runs aditional transformations for cleaning the dataset.

    Parameters
    ----------
    df: DataFrame
        DataFrame with messages to be cleaned

    Returns
    -------
        df: DataFrame
            Cleaned DataFrame
    """
    return df.drop_duplicates()


def save_data(df, database_filename):
    """Stores the processed message's dataframe

    Parameters
    ----------
    df: DataFrame
        DataFrame to store.

    database_filename: string
        Name of the database to store the data.

    """
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
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
