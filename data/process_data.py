import sys
import pandas as pd
import sys
import os
import click

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_utils import DataUtils


dataUtils = DataUtils()

@click.command(help="Run ETL pipeline that cleans data and stores in database")
@click.option("--disaster_messages")
@click.option("--disaster_categories")
@click.option("--database_path")
def process_data(disaster_messages, disaster_categories, database_path):

    df = dataUtils.load_data(disaster_messages, disaster_categories)

    print('Cleaning data...')
    df = dataUtils.clean_data(df)

    print('Saving data...\n    DATABASE: {}'.format(database_path))
    dataUtils.save_db_data(df, database_path)

    print('Cleaned data saved to database!')


if __name__ == '__main__':
    process_data()
