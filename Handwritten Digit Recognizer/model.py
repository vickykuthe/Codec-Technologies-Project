import cv2
import numpy as np


class KNNModel:

    def __init__(self, k=3):
        """
        Create KNN model.
        """
        self.k = k

        self.model = cv2.ml.KNearest_create()

    def train(self, features, labels):
        """
        Train the KNN model.
        """

        features = np.asarray(
            features,
            dtype=np.float32
        )

        labels = np.asarray(
            labels,
            dtype=np.float32
        )

        self.model.train(
            features,
            cv2.ml.ROW_SAMPLE,
            labels
        )

    def predict(self, features):
        """
        Predict digit classes.
        """

        features = np.asarray(
            features,
            dtype=np.float32
        )

        _, results, _, _ = self.model.findNearest(
            features,
            self.k
        )

        return results.ravel()