import sys
import os
import click
import scipy.sparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_utils import DataUtils
from model_utils import ModelUtils
from nlp_utils import NLPUtils


modelUtils = ModelUtils()
dataUtils = DataUtils()
nlpUtils = NLPUtils()

@click.command(help="Run ETL pipeline for feature engineering")
@click.option("--database_path")
@click.option("--sample")
@click.option("--model_features_path")
@click.option("--data_vector_path")
@click.option("--feature_matrix")
@click.option("--target_dataset")
def etl_data(database_path, sample, model_features_path, data_vector_path, feature_matrix, target_dataset):
    X, y, category_names = dataUtils.load_db_data(database_path, sample)    
    y.to_csv(target_dataset, index = False)

    X = nlpUtils.create_vector_model(X)
    X = nlpUtils.clean_count_vector(X, model_features_path)
    X = nlpUtils.normalize_count_vector(X, data_vector_path)
    scipy.sparse.save_npz(feature_matrix, X)

if __name__ == '__main__':
    etl_data()