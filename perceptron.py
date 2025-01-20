`"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from data import make_dataset
from plot import plot_boundary
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV


# (Question 3): Perceptron


class PerceptronClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_iter=5, learning_rate=.0001):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.classes_ = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(np.clip(-z, -200, 200)))

    def fit(self, X, y):
        """Fit a perceptron model on (X, y)

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """

        # Input validation
        X = np.asarray(X, dtype=np.float64)
        n_instances, n_features = X.shape

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        n_classes = len(np.unique(y))
        if n_classes != 2:
            raise ValueError("This class is only dealing with binary "
                             "classification problems")

        self.classes_ = np.unique(y)
        X_bias = np.column_stack([np.ones(n_instances), X])
        indices = np.arange(n_instances)
        self.weights = np.zeros(n_features + 1)

        # n_iter epochs
        for _ in range(self.n_iter):
            np.random.shuffle(indices)
            X_shuffled = X_bias[indices]
            y_shuffled = y[indices]

            for i in range(n_instances):
                # Forward
                z = np.dot(X_shuffled[i], self.weights)
                # Gradient descent
                gradient = (self.sigmoid(z) - y_shuffled[i]) * X_shuffled[i]
                self.weights -= self.learning_rate * gradient

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

        X = np.asarray(X, dtype=np.float64)
        n_instances, n_features = X.shape
        X_bias = np.column_stack([np.ones(n_instances), X])

        proba_1 = self.sigmoid(np.dot(X_bias, self.weights))

        return np.column_stack([1 - proba_1, proba_1])

def perceptron():
    X, y = make_dataset(3000)
    X_train = X[:1000]
    y_train = y[:1000]
    X_test  = X[-2000:]
    y_test  = y[-2000:]

    learning_rates = [1e-4, 5e-4, 1e-3, 1e-2, 1e-1]
    n_epochs = 5

    print("===== Perceptron =====")
    for current_lr in learning_rates:
        classifier = PerceptronClassifier(n_iter=n_epochs, learning_rate=current_lr)
        classifier.fit(X_train, y_train)

        title = f"Perceptron Decision Boundary\nLearning Rate = {current_lr:.4f}, Epochs = {n_epochs}"
        file = f"perceptron_learning_rate_{current_lr:.4f}"
        plot_boundary(file, classifier, X_test, y_test, title=title)

def test_accuracy():
    learning_rates = [1e-4, 5e-4, 1e-3, 1e-2, 1e-1]
    n_epochs = 5
    n_runs = 5
    accuracies = np.zeros((len(learning_rates), n_runs))

    for run in range(n_runs):
        X, y = make_dataset(3000)
        X_train = X[:1000]
        y_train = y[:1000]
        X_test  = X[-2000:]
        y_test  = y[-2000:]

        for i, current_lr in enumerate(learning_rates):
            classifier = PerceptronClassifier(n_iter=n_epochs, learning_rate=current_lr)
            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(X_test)
            accuracies[i, run] = accuracy_score(y_test, y_pred)

    print("===== Accuracy =====")
    print("Accuracy for each test (columns), for each hyperparameter value (rows) :")
    print(accuracies)
    print("Mean of the accuracy for each hyperparameter value :")
    mean_accuracies = np.mean(accuracies, axis=1)
    print(mean_accuracies)
    print("Standard deviation of the accuracies for each hyperparameter value :")
    std_accuracies = np.std(accuracies, axis=1)
    print(std_accuracies)

def tuning():
    X, y = make_dataset(3000)
    X_train = X[:1000]
    y_train = y[:1000]
    X_test  = X[-2000:]
    y_test  = y[-2000:]

    X_noisy, y_noisy = make_dataset(3000, n_irrelevant=200)
    X_noisy_train = X_noisy[:1000]
    y_noisy_train = y_noisy[:1000]
    X_noisy_test  = X_noisy[-2000:]
    y_noisy_test  = y_noisy[-2000:]

    param_grid = {
        'learning_rate': [1e-4, 5e-4, 1e-3, 1e-2, 1e-1, 5e-1]
    }
    perceptron = PerceptronClassifier()
    perceptron_noisy = PerceptronClassifier()
    grid_search = GridSearchCV(
        estimator=perceptron,
        param_grid=param_grid,
        scoring='accuracy',
        verbose=1
    )
    grid_search_noisy = GridSearchCV(
        estimator=perceptron_noisy,
        param_grid=param_grid,
        scoring='accuracy',
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    grid_search_noisy.fit(X_noisy_train, y_noisy_train)
    print("===== Tuning =====")
    print(f"Best result : [param: {grid_search.best_params_}, accuracy :{grid_search.best_score_}] from the tried paramaters {param_grid['learning_rate']}")
    print(f"Best result with noisy dataset: [param: {grid_search_noisy.best_params_}, accuracy :{grid_search_noisy.best_score_}] from the tried paramaters {param_grid['learning_rate']}")

    n_runs = 10
    accuracies = np.zeros(n_runs)
    accuracies_noisy = np.zeros(n_runs)
    for run in range(n_runs):
        X, y = make_dataset(3000)
        X_train = X[:1000]
        y_train = y[:1000]
        X_test  = X[-2000:]
        y_test  = y[-2000:]

        X_noisy, y_noisy = make_dataset(3000, n_irrelevant=200)
        X_noisy_train = X_noisy[:1000]
        y_noisy_train = y_noisy[:1000]
        X_noisy_test  = X_noisy[-2000:]
        y_noisy_test  = y_noisy[-2000:]

        classifier = PerceptronClassifier(learning_rate=grid_search.best_params_['learning_rate'])
        classifier_noisy = PerceptronClassifier(learning_rate=grid_search_noisy.best_params_['learning_rate'])

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracies[run] = accuracy_score(y_test, y_pred)

        classifier_noisy.fit(X_noisy_train, y_noisy_train)
        y_noisy_pred = classifier_noisy.predict(X_noisy_test)
        accuracies_noisy[run] = accuracy_score(y_noisy_test, y_noisy_pred)

    print("* Non-noisy results")
    print("Accuracy for each test :")
    print(accuracies)
    print("Mean of the accuracy :")
    print(f"{np.mean(accuracies):.4f}")
    print("Standard deviation of the accuracies for each hyperparameter value :")
    print(f"{np.std(accuracies):.5f}")

    print("* Noisy results")
    print("Accuracy for each test :")
    print(accuracies_noisy)
    print("Mean of the accuracy :")
    print(f"{np.mean(accuracies_noisy):.4f}")
    print("Standard deviation of the accuracies for each hyperparameter value :")
    print(f"{np.std(accuracies_noisy):.5f}")

if __name__ == "__main__":
    seed = int(ord('<')+ord('3'))
    np.random.seed(seed)
    perceptron()
    test_accuracy()
    tuning()

    
