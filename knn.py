"""
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

# (Question 2): KNN

# 1.(a)
def knn():
    """Generate a dataset and fit multiples KNN models with different values
        for K : 1, 5, 50, 100, 500.
        Print the accuracy and confusion matrix for each model
        as well as produce a plot of the decision boundary.
    """
    X, y = make_dataset(3000)
    X_train = X[:1000]
    y_train = y[:1000]
    X_test  = X[-2000:]
    y_test  = y[-2000:]
    n_neighbors = [1,5,50,100,500]
    print("===== Decision tree =====")
    for k_value in n_neighbors:
        knn = KNeighborsClassifier(n_neighbors=k_value)
        knn_fit = knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        print("Accuracy: ", accuracy_score(y_test, y_pred))
        print("Confusion matrix: ", confusion_matrix(y_test, y_pred))
        plot_boundary("nearest_neighbors_" + str(k_value), knn_fit, X, y,
                      title="K-Nearest neighbors where K is " + str(k_value))
    return

# 2
def test_accuracy():
    """Calculate and print the average test set accuracies over 
        5 runs for each value of max_depth : 1, 2, 4, 8 and None.
        As well as its standard deviation per run.
    """
    n_neighbors = [1,5,50,100,500]
    n_runs = 5
    accuracies = np.zeros((len(n_neighbors), n_runs))
    for run in range(n_runs):
        X, y = make_dataset(3000)
        X_train = X[:1000]
        y_train = y[:1000]
        X_test  = X[-2000:]
        y_test  = y[-2000:]

        for i, current_k in enumerate(n_neighbors):
            classifier = KNeighborsClassifier(n_neighbors=current_k)
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
    return

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
        'n_neighbors': [1,5,50,100,500]
    }
    
    knn = KNeighborsClassifier()
    knn_noisy = KNeighborsClassifier()
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        scoring='accuracy',
        verbose=1
    )
    grid_search_noisy = GridSearchCV(
        estimator=knn_noisy,
        param_grid=param_grid,
        scoring='accuracy',
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    grid_search_noisy.fit(X_train, y_train)
    print("===== Tuning =====")
    print(f"Best result : [param: {grid_search.best_params_}, accuracy :{grid_search.best_score_}] from the tried paramaters {param_grid["n_neighbors"]}")
    print(f"Best result with noisy dataset : [param: {grid_search_noisy.best_params_}, accuracy :{grid_search_noisy.best_score_}] from the tried paramaters {param_grid["n_neighbors"]}")

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

        classifier = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'])
        classifier_noisy = KNeighborsClassifier(n_neighbors=grid_search_noisy.best_params_['n_neighbors'])

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

def tuning2():
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
        'n_neighbors': [1,5,50,100,500]
    }
    
    knn = KNeighborsClassifier()
    knn_noisy = KNeighborsClassifier()
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        scoring='accuracy',
        verbose=1
    )
    grid_search_noisy = GridSearchCV(
        estimator=knn_noisy,
        param_grid=param_grid,
        scoring='accuracy',
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    grid_search_noisy.fit(X_noisy_train, y_noisy_train)
    print("===== Tuning =====")
    print(f"Best result : [param: {grid_search.best_params_}, accuracy :{grid_search.best_score_}] from the tried paramaters {param_grid['n_neighbors']}")
    print(f"Best result with noisy dataset: [param: {grid_search_noisy.best_params_}, accuracy :{grid_search_noisy.best_score_}] from the tried paramaters {param_grid['n_neighbors']}")

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

        classifier = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'])
        classifier_noisy = KNeighborsClassifier(n_neighbors=grid_search_noisy.best_params_['n_neighbors'])

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
    knn()
    test_accuracy()
    tuning()