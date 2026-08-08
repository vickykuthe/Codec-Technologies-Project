import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(data):
    """
    Clean and prepare the stock dataset.
    """

    data = data.copy()

    # Convert numerical columns to numbers
    numerical_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for column in numerical_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Remove missing values
    data = data.dropna().reset_index(drop=True)

    # Convert Date column
    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    # Remove invalid dates
    data = data.dropna(
        subset=["Date"]
    )

    # Sort data chronologically
    data = data.sort_values(
        "Date"
    ).reset_index(drop=True)

    return data


def normalize_data(data):
    """
    Normalize stock features between 0 and 1.
    """

    features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    scaler = MinMaxScaler()

    scaled_values = scaler.fit_transform(
        data[features]
    )

    scaled_data = pd.DataFrame(
        scaled_values,
        columns=features,
        index=data.index
    )

    return scaled_data, scaler