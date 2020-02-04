import pandas as pd
from sqlalchemy import create_engine

class DataUtils():

    def load_db_data(self, database_filepath):
        """ Creates the dataset from a database.

        Parameters
        ----------
        database_filepath: string
            Path to the database for importing the data.

        X: DataFrame
            Dataset features.
        y: DataFrame
            Dataset targets (Categories).

        category_names: list
            List of category names.
        """
        print('sqlite:///'+database_filepath)
        engine = create_engine('sqlite:///'+database_filepath)
        df = pd.read_sql('messages', engine).sample(1000)
        X = df[['message']].values.flatten()
        y = df.drop(['id', 'message', 'original'], axis = 1)
        category_names = list(y.columns.values)

        return X, y, category_names

    def get_encodings(self,filepath):
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

    def load_data(self, messages_filepath, categories_filepath):
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

    def clean_data(self, df):
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


    def save_db_data(self, df, database_filename):
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