import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_basic_lstm(
    sequence_length,
    number_of_features
):

    model = Sequential()

    model.add(
        LSTM(
            64,
            return_sequences=True,
            input_shape=(
                sequence_length,
                number_of_features
            )
        )
    )

    model.add(
        LSTM(
            32,
            return_sequences=False
        )
    )

    model.add(
        Dense(16, activation="relu")
    )

    model.add(
        Dense(1)
    )

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model


def build_improved_lstm(
    sequence_length,
    number_of_features
):

    model = Sequential()

    model.add(
        LSTM(
            128,
            return_sequences=True,
            input_shape=(
                sequence_length,
                number_of_features
            )
        )
    )

    model.add(
        Dropout(0.2)
    )

    model.add(
        LSTM(
            64,
            return_sequences=False
        )
    )

    model.add(
        Dropout(0.2)
    )

    model.add(
        Dense(
            32,
            activation="relu"
        )
    )

    model.add(
        Dense(1)
    )

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model