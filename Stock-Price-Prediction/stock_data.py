import numpy as np


def create_sequences(data, sequence_length=50):
    """
    Create sequences for the LSTM model.

    The model uses the previous 50 trading days
    to predict the next day's closing price.
    """

    data_array = data.to_numpy(dtype=np.float32)

    # Find the position of the Close column
    close_index = data.columns.get_loc("Close")

    X = []
    y = []

    for i in range(sequence_length, len(data_array)):

        # Previous 50 trading days
        X.append(
            data_array[i - sequence_length:i]
        )

        # Next day's closing price
        y.append(
            data_array[i, close_index]
        )

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y