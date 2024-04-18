from typing import List, Union

import joblib
import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from logging import getLogger
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

TEST_SIZE = 0.2
N_EPOCH_DOC2VEC = 100
VECTOR_SIZE_DOC2VEC = 125
LABELS = [
    'Politics' ,
    'Health' ,
    'Finance' ,
    'Travel' ,
    'Food' ,
    'Education' ,
    'Environment',
    'Fashion' ,
    'Science' ,
    'Sports' ,
    'Technology' ,
    'Entertainment'
]
DEFAULT_PARAM_GRID = {
    "svc": {"kernel": ["linear", "rbf", "poly"], "C": [0.1, 1, 10, 100]},
    "logreg": {"C": [100, 10, 1.0, 0.1, 0.01], "penalty": ["l2", "l1"], "solver": ["liblinear"]},
    "randomforest": {"n_estimators": [5,10,20,50,100], "max_depth": [5,10,20,50,100]},
}

logger = getLogger(__name__)

class TopicClassifier:
    "Class for intent classification using a model for vectorization followed by a ML Classifier."

    def __init__(self, classifier_name: str = "svc", vectorizer_name: str = "tfidf"):
        self.vectorizer_name = vectorizer_name
        if vectorizer_name == "tfidf":
            self.vectorizer = TfidfVectorizer()
        elif vectorizer_name == "count":
            self.vectorizer = CountVectorizer()
        elif vectorizer_name == "doc2vec":
            self.vectorizer = Doc2Vec(vector_size=VECTOR_SIZE_DOC2VEC, epochs=N_EPOCH_DOC2VEC)
        else:
            logger.error(
                f"Vectorizer {vectorizer_name} not implemented yet. Please choose between tfidf, count and doc2vec."
            )
            exit()
        self.classifier_name = classifier_name
        if self.classifier_name == "svc":
            self.classifier = SVC()
        elif self.classifier_name == "logreg":
            self.classifier = LogisticRegression()
        elif self.classifier_name == "randomforest":
            self.classifier = RandomForestClassifier()
        else:
            logger.error(
                f"Classifier {self.classifier_name} not implemented yet. Please choose between svc and logreg."
            )
            exit()

    def perform_grid_search_(
        self, X_train: np.ndarray, y_train: np.ndarray, param_grid: dict 
    ) -> dict:
        """Grid search to find the best hyperparameters for the classifier. Uses 5-fold cross validation.

        Args:
            X_train (np.ndarray): Array of embeddings of the training data.
            y_train (np.ndarray): Labels of the training data.
            param_grid (dict, optional): Dictionnary mapping classifier names to the list
                                            and values of hyperparameters to tune.
                                            Defaults to DEFAULT_PARAM_GRID.

        Returns:
            dict: Best hypeparameters found by the grid search
        """
        grid_search = GridSearchCV(
            self.classifier, param_grid[self.classifier_name], cv=5, scoring="f1_weighted", n_jobs=-1, verbose=2
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        return best_params

    def fit_transform_vectorizer(self, X_train: List[str]) -> np.ndarray:
        """Method to fit the vectorizer on the training data.

        Args:
            X_train (List[str]): List of verbatims composing the training data.
        """
        if self.vectorizer_name in ["count", "tfidf"]:
            return self.vectorizer.fit_transform(X_train)
        elif self.vectorizer_name == "doc2vec":
            tokens_sentences = [sentence.split() for sentence in X_train]
            tagged_data = [TaggedDocument(words=words, tags=[str(i)]) for i, words in enumerate(tokens_sentences)]
            self.vectorizer.build_vocab(tagged_data)
            self.vectorizer.train(
                tagged_data, total_examples=self.vectorizer.corpus_count, epochs=self.vectorizer.epochs
            )
            return [self.vectorizer.infer_vector(tokens_sentence) for tokens_sentence in tokens_sentences]
        else:
            logger.error(
                f"Vectorizer {self.vectorizer_name} not implemented yet. Please choose between tfidf, count & doc2vec."
            )
            exit()

    def transform_vectorizer(self, X: List[str]) -> np.ndarray:
        """Method to transform new data with the vectorizer.

        Args:
            X (List[str]): List of verbatims to transform.

        Returns:
            np.ndarray: Array of embeddings of the verbatims.
        """
        if self.vectorizer_name in ["count", "tfidf"]:
            return self.vectorizer.transform(X)
        elif self.vectorizer_name == "doc2vec":
            tokens_sentences = [sentence.split() for sentence in X]
            return [self.vectorizer.infer_vector(tokens_sentence) for tokens_sentence in tokens_sentences]
        else:
            logger.error(
                f"Vectorizer {self.vectorizer_name} not implemented yet. Please choose between tfidf, count & doc2vec."
            )
            exit()

    def train_classifier(
        self,
        X_train: np.ndarray,
        y_train: List[str],
        tune_hyperparameters: bool = True,
    ):
        """Method to train the intent classifier.

        Args:
            X_train (np.ndarray): Array of embeddings of the training data.
            y_train (List[str]): Labels of the training data.
            tune_hyperparameters (bool, optional): Whether to finetune the hyperparameters of the model.
                                                    Defaults to True.
            class_weights (dict, optinal): Particular weights given to the different labels. Default to CLASS_WEIGHTS.
        """
        if tune_hyperparameters:
            best_params = self.perform_grid_search_(X_train, y_train, DEFAULT_PARAM_GRID)
            for param in best_params:
                setattr(self.classifier, param, best_params[param])
            self.classifier.fit(X_train, y_train)
        else:
            self.classifier.fit(X_train, y_train)

    def get_classifier_parameters(self) -> dict:
        "Method to get the parameters of the classifier."
        return self.classifier.get_params()

    def evaluate_classifier(
        self, X_test: np.ndarray, y_test: List[str]
    ) -> Union[List[str], str, np.ndarray, float]:
        """Methode to evaluate the trained intent classifier on the test set.

        Args:
            X_test (np.ndarray): Array of embeddings of the test data.
            y_test (List[str]): Labels of the test data.

        Returns:
            Union[List[str], str, np.ndarray]: Predicted labels, classification report, confusion matrix
            and custom weighted f1-score (weighted by CLASS_WEIGHTS).
        """
        y_pred = self.classifier.predict(X_test)
        report = classification_report(y_test, y_pred, zero_division=np.nan, output_dict=True)
        accuracy_score = round(report["accuracy"], 2)
        

        logger.info("Accuracy: " + str(accuracy_score))
        conf_matrix = confusion_matrix(y_test, y_pred, labels=LABELS)
        return (
            y_pred,
            classification_report(y_test, y_pred, zero_division=np.nan),
            conf_matrix
        )  

    def inference(self, X: np.ndarray) -> List[str]:
        "Method to perform inference on new data X (list of string sentences)."
        y_pred = self.classifier.predict(X)
        return y_pred

    def from_pretrained(self, classifier_path: str, vectorizer_path: str):
        """Load a pretrained IntentClassifier (classier + vectorizer).

        Args:
            classifier_path (str): Path where the classifier is stored.
            vectorizer_path (str): Path where the vectorizer is stored.
        """
        self.classifier = joblib.load(classifier_path)
        if self.vectorizer_name in ["tfidf", "count"]:
            self.vectorizer = joblib.load(vectorizer_path)
        elif self.vectorizer_name == "doc2vec":
            self.vectorizer = Doc2Vec.load(vectorizer_path)
        else:
            logger.error(
                f"Vectorizer {self.vectorizer_name} not implemented yet. Please choose between tfidf, count & doc2vec."
            )
            exit()

    def save_model_and_vectorizer(self, classifier_path, vectorizer_path):
        """Save the classifier and vectorizer at the specified paths.

        Args:
            classifier_path (str): Path where the classifier is stored.
            vectorizer_path (str): Path where the vectorizer is stored."""
        joblib.dump(self.classifier, classifier_path + ".pkl")
        if self.vectorizer_name in ["tfidf", "count"]:
            joblib.dump(self.vectorizer, vectorizer_path + ".pkl")
        elif self.vectorizer_name == "doc2vec":
            self.vectorizer.save(vectorizer_path + ".bin")
        else:
            logger.error(
                f"Vectorizer {self.vectorizer_name} not implemented yet. Please choose between tfidf, count & doc2vec."
            )
            exit()
