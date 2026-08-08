import pandas as pd


def load_stock_data(file_path):
    data = pd.read_csv(file_path)

    print("Stock data loaded successfully.")
    print("Dataset shape:", data.shape)

    return data