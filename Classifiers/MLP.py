from skmultiflow.core import BaseSKMObject, ClassifierMixin
from sklearn.neural_network import MLPClassifier as MLP
import numpy as np


class MLPClassifier(BaseSKMObject, ClassifierMixin):
    def __init__(self, hidden_layer_sizes):
        super().__init__()
        # initialize MLP classifier
        self.classifier = MLP(hidden_layer_sizes=hidden_layer_sizes)

    def fit(self, X, y, classes=None, sample_weight=None):
        """ partial_fit
        Calls the Perceptron partial_fit from sklearn.
        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The feature's matrix.
        y: Array-like
            The class labels for all samples in X.
        classes: Not used.
        sample_weight:
            Samples weight. If not provided, uniform weights are assumed.
        Returns
        -------
        MLPClassifier
            self
        """
        self.classifier.fit(X=X, y=y)
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Calls the Perceptron fit function from sklearn.
               Parameters
               ----------
               X: numpy.ndarray of shape (n_samples, n_features)
                   The feature's matrix.
               y: Array-like
                   The class labels for all samples in X.
               classes: Not used.
               sample_weight:
                   Samples weight. If not provided, uniform weights are assumed.
               Returns
               -------
               PerceptronMask
                   self
        """
        self.classifier.partial_fit(X=X, y=y, classes=classes)
        return self

    def predict(self, X):
        """ predict
        Uses the current model to predict samples in X.
        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The feature's matrix.
        Returns
        -------
        numpy.ndarray
            A numpy.ndarray containing the predicted labels for all instances in X.
        """
        return self.classifier.predict(X)

    def predict_proba(self, X):
        """ Predicts the probability of each sample belonging to each one of the known classes.

                Parameters
                ----------
                X: Numpy.ndarray of shape (n_samples, n_features)
                    A matrix of the samples we want to predict.

                Returns
                -------
                numpy.ndarray
                    An array of shape (n_samples, n_features), in which each outer entry is
                    associated with the X entry of the same index. And where the list in
                    index [i] contains len(self.target_values) elements, each of which represents
                    the probability that the i-th sample of X belongs to a certain label.

        """
        return self.classifier.predict_proba(X)
