import pandas as pd

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, make_scorer, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV


from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from nlp_utils import NLPUtils

class ModelUtils():

    nlpUtils = NLPUtils()

    def build_model(self):        
        """ Builds the pipeline and finds the best classification model with gridsearch.

        Returns
        -------

        cv: GridSearchCV
            GridSearchCV instance with the tuned model.
        """
        forest = RandomForestClassifier()
        pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=self.nlpUtils.tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(forest))
        ])

        parameters = {
        }

        scorer = make_scorer(f1_score)
        cv = GridSearchCV(pipeline, scoring=scorer, param_grid=parameters, verbose=50)

        return cv


    def evaluate_model(self, model, X_test, Y_test, category_names):
        """Model evaluation

        Parameters
        ----------

        model: GridSearchCV
            GridSearchCV instance with the tuned model.

        X_test: Series.
            Dataset with the test features (messages).

        Y_test: Series.
            Dataset with the test targets (categories).
        """
        y_pred = model.predict(X_test)
        y_pred = pd.DataFrame(data = y_pred, columns = category_names)
        for category in category_names:
            print("Scoring %s"%(category))
            print(classification_report(Y_test[category].values, y_pred[category].astype(int)))


    def save_model(self,model, model_filepath):
        """Stores model

        Parameters
        ----------

        model: GridSearchCV
            GridSearchCV instance with the tuned model.

        model_filepath: string
            Path to store the model
        """
        import pickle
        # save the classifier
        pickle.dump(model, open(model_filepath, 'wb'))
        return True