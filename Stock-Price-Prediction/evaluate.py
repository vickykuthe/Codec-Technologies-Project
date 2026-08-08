import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_metrics(actual, predicted):
    """
    Calculate MSE, RMSE and MAE.
    """

    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()

    mse = mean_squared_error(
        actual,
        predicted
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        actual,
        predicted
    )

    return mse, rmse, mae