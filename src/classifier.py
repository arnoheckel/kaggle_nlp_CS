from torch import nn
import torch
import pandas as pd
import numpy as np
from scipy.special import softmax
from typing import List, Union

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

DEFAULT_PARAM_GRID = {
    "svc": {"kernel": ["linear", "rbf", "poly"], "C": [0.1, 1, 10, 100]},
    "logreg": {"C": [100, 10, 1.0, 0.1, 0.01], "penalty": ["l2", "l1"], "solver": ["liblinear"]},
    "randomforest": {"n_estimators": [5,10,20,50,100], "max_depth": [5,10,20,50,100]},
}

class DNNClassifier(nn.Module):
    def __init__(self,n_classes:int,input_size:int=768):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.linear1 = nn.Linear(input_size,64)
        self.ReLu = nn.ReLU()
        self.linear2 = nn.Linear(64,n_classes)

    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.ReLu(x)
        logits = self.linear2(x)
        return logits
    
    def predict(self, X: np.ndarray) -> list  :
        """Method to perform inference on new data X (list of string sentences
        Returns the predicted labels
        """
        self.eval()
        with torch.no_grad():
            X = torch.tensor(X).float()
            logits = self.forward(X)
            predictions = torch.argmax(logits, dim=1)
        return predictions.numpy()
    
    def evaluate_classifier(
        self, X_test: np.ndarray, y_test: list
    ):
        """Methode to evaluate the trained classifier on the test set.

        Args:
            X_test (np.ndarray): Array of embeddings of the test data.
            y_test (List[str]): Labels of the test data.

        Returns:
            Union[List[str], str, np.ndarray]: Predicted labels, classification report, confusion matrix
            and custom weighted f1-score (weighted by CLASS_WEIGHTS).
        """
        self.eval()
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred, zero_division=np.nan, output_dict=True)
        accuracy_score = round(report["accuracy"], 2)
        print("Accuracy: " + str(accuracy_score))

        conf_matrix = confusion_matrix(y_test, y_pred)
        return (
            y_pred,
            classification_report(y_test, y_pred, zero_division=np.nan),
            conf_matrix
        )
    
    
class FewShotClassifier:
    """
    This class is a few-shot classifier.
    Pass the few_shot_data as a pd.DataFrame with columns 'embedding','label'
    """

    def __init__(self,X_ref:np.ndarray,y_ref:List[str],distance_metric:str ='cosine',nb_instances: int=1):
        self.ref_embedding = X_ref
        self.ref_label = y_ref
        assert distance_metric in ['cosine','euclidean'] , "Distance metric should be either 'cosine' or 'euclidean'"
        self.distance_metric = distance_metric
        self.nb_instances = nb_instances


    def predict(self, embeddings:List[np.ndarray]):
        """
        This function takes an embedding matrix as input and returns the predicted labels for each row.
        """


        if self.distance_metric == 'cosine':
            distance_mat = 1 - cosine_similarity(embeddings,self.ref_embedding)
        else:
            distance_mat = euclidean_distances(embeddings,self.ref_embedding)
        
        probabilities = []
        predictions = []
        for i in range(len(embeddings)):
            distances = list(distance_mat[i])
            distances_df = pd.DataFrame({'distance':distances,'label':list(self.ref_label)})
            if not self.nb_instances:
                label_distance = distances_df.groupby('label',as_index=False).agg({'distance':'mean'})
            else:
                distances_df.sort_values('distance',inplace=True)
                label_distance = distances_df.groupby('label',as_index=False).head(self.nb_instances)
                label_distance = label_distance.groupby('label',as_index=False).agg({'distance':'mean'})
            label_distance.sort_values('distance',inplace=True)
            probabilities.append(dict(zip(label_distance['label'],softmax(label_distance['distance']))))
            predictions.append(label_distance['label'].values[0])

        return predictions,probabilities
    
    def evaluate_classifier(
        self, X_test: np.ndarray, y_test: List[str]
    ) -> Union[List[str], str, np.ndarray, float]:
        """Methode to evaluate the trained classifier on the test set.

        Args:
            X_test (np.ndarray): Array of embeddings of the test data.
            y_test (List[str]): Labels of the test data.

        Returns:
            Union[List[str], str, np.ndarray]: Predicted labels, classification report, confusion matrix
            and custom weighted f1-score (weighted by CLASS_WEIGHTS).
        """
        y_pred = self.predict(X_test)[0]
        report = classification_report(y_test, y_pred, zero_division=np.nan, output_dict=True)
        accuracy_score = round(report["accuracy"], 2)
        print("Accuracy: " + str(accuracy_score))

        conf_matrix = confusion_matrix(y_test, y_pred)
        return (
            y_pred,
            classification_report(y_test, y_pred, zero_division=np.nan),
            conf_matrix
        )  
    
    
class MLClassifier:
    "Class for intent classification using a model for vectorization followed by a ML Classifier."

    def __init__(self, classifier_name: str = "svc"):

        self.classifier_name = classifier_name
        if self.classifier_name == "svc":
            self.classifier = SVC()
        elif self.classifier_name == "logreg":
            self.classifier = LogisticRegression()
        elif self.classifier_name == "randomforest":
            self.classifier = RandomForestClassifier()
        else:
            print(f"Classifier {self.classifier_name} not implemented yet. Please choose between svc, logreg and randomforest.")


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

    def predict(self, X: np.ndarray) -> List[str]:
        "Method to perform inference on new data X (list of string sentences)."
        y_pred = self.classifier.predict(X)
        return y_pred

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
        print("Accuracy: " + str(accuracy_score))

        conf_matrix = confusion_matrix(y_test, y_pred)
        return (
            y_pred,
            classification_report(y_test, y_pred, zero_division=np.nan),
            conf_matrix
        )  