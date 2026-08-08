import numpy as np
from sklearn.linear_model import LinearRegression


def train_linear_regression(X_train, y_train):

    model = LinearRegression()

    X_train = np.asarray(X_train).reshape(-1, 1)
    y_train = np.asarray(y_train).reshape(-1, 1)

    model.fit(X_train, y_train)

    return model


def predict_linear_regression(model, X_test):

    X_test = np.asarray(X_test).reshape(-1, 1)

    predictions = model.predict(X_test)

    return predictions.ravel()