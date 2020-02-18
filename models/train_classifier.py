import sys
import os
import pandas as pd
import scipy.sparse
import click

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_utils import DataUtils
from model_utils import ModelUtils

modelUtils = ModelUtils()
dataUtils = DataUtils()

@click.command("Builds the Machine Learning pipeline that will be used to predict the categories in which a given message belongs to.")
@click.option("--database_path")
@click.option("--feature_matrix")
@click.option("--target_dataset")
@click.option("--model_filepath")
@click.option("--test_size")
def train_model(database_path, feature_matrix, target_dataset, model_filepath):
    X = scipy.sparse.load_npz('/tmp/sparse_matrix.npz')
    y = pd.read_csv(target_dataset)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = MultiOutputClassifier(RandomForestClassifier())
    model.fit(X_train, y_train)
    scores = modelUtils.evaluate_model(model, X_test, y_test, list(y.columns.values))
    modelUtils.save_model(model, model_filepath)


if __name__ == '__main__':
    train_model()
