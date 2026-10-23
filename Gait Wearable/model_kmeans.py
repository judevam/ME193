"""k-means used as a classifier.

k-means groups windows into clusters without looking at the labels. To classify with it,
each cluster is named after the activity most of its training windows came from, and a new
window gets the name of the nearest cluster center. Features are standardized first, so
big-valued features (gyro in deg/s) don't drown out small ones (accel in g).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class KMeansClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clusters_per_class: int = 2, random_state: int = 0):
        self.clusters_per_class = clusters_per_class
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        k = min(len(X), self.clusters_per_class * len(self.classes_))
        self.kmeans_ = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(Xs)
        # Name each cluster after its most common activity
        self.cluster_labels_ = np.empty(k, dtype=object)
        for c in range(k):
            members = y[self.kmeans_.labels_ == c]
            self.cluster_labels_[c] = Counter(members).most_common(1)[0][0] if len(members) else self.classes_[0]
        return self

    def predict(self, X):
        return self.cluster_labels_[self.kmeans_.predict(self.scaler_.transform(np.asarray(X, float)))]
