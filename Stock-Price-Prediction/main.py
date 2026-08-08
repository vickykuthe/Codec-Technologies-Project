import numpy as np

from data_loader import load_stock_data
from preprocess import preprocess_data, normalize_data
from stock_data import create_sequences
from lstm_model import build_basic_lstm, build_improved_lstm
from evaluate import calculate_metrics
from visualize import (
    plot_stock_price,
    plot_predictions,
    plot_training_history
)


# ==========================================
# SETTINGS
# ==========================================

DATA_FILE = "google.csv"

SEQUENCE_LENGTH = 50

TRAIN_RATIO = 0.80

EPOCHS = 10

BATCH_SIZE = 32


# ==========================================
# LOAD DATA
# ==========================================

print("\nLoading stock data...")

data = load_stock_data(
    DATA_FILE
)


# ==========================================
# PREPROCESS DATA
# ==========================================

print("\nPreprocessing data...")

data = preprocess_data(
    data
)

print(
    "Processed dataset:",
    data.shape
)


# ==========================================
# DISPLAY STOCK PRICE
# ==========================================

plot_stock_price(
    data
)


# ==========================================
# NORMALIZE DATA
# ==========================================

print("\nNormalizing data...")

scaled_data, scaler = normalize_data(
    data
)


# ==========================================
# CREATE SEQUENCES
# ==========================================

print("\nCreating LSTM sequences...")

X, y = create_sequences(
    scaled_data,
    SEQUENCE_LENGTH
)

print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

split_index = int(
    len(X) * TRAIN_RATIO
)

X_train = X[:split_index]

X_test = X[split_index:]

y_train = y[:split_index]

y_test = y[split_index:]


print("\nTraining data:")
print(
    "X_train:",
    X_train.shape
)

print(
    "y_train:",
    y_train.shape
)

print("\nTesting data:")

print(
    "X_test:",
    X_test.shape
)

print(
    "y_test:",
    y_test.shape
)


# ==========================================
# BASIC LSTM
# ==========================================

print("\n")
print("=" * 50)
print("TRAINING BASIC LSTM")
print("=" * 50)


basic_model = build_basic_lstm(
    SEQUENCE_LENGTH,
    X_train.shape[2]
)

basic_model.summary()


basic_history = basic_model.fit(
    X_train,
    y_train,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    validation_split=0.1,

    verbose=1
)


# ==========================================
# BASIC LSTM PREDICTION
# ==========================================

print("\nPredicting using Basic LSTM...")

basic_predictions = basic_model.predict(
    X_test,
    verbose=0
).flatten()


# ==========================================
# BASIC LSTM EVALUATION
# ==========================================

basic_mse, basic_rmse, basic_mae = calculate_metrics(
    y_test,
    basic_predictions
)


print("\nBasic LSTM Results")

print(
    "MSE :",
    basic_mse
)

print(
    "RMSE:",
    basic_rmse
)

print(
    "MAE :",
    basic_mae
)


plot_training_history(
    basic_history
)

plot_predictions(
    y_test,
    basic_predictions,
    "Basic LSTM - Actual vs Predicted"
)


# ==========================================
# IMPROVED LSTM
# ==========================================

print("\n")
print("=" * 50)
print("TRAINING IMPROVED LSTM")
print("=" * 50)


improved_model = build_improved_lstm(
    SEQUENCE_LENGTH,
    X_train.shape[2]
)

improved_model.summary()


improved_history = improved_model.fit(
    X_train,
    y_train,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    validation_split=0.1,

    verbose=1
)


# ==========================================
# IMPROVED LSTM PREDICTION
# ==========================================

print("\nPredicting using Improved LSTM...")

improved_predictions = improved_model.predict(
    X_test,
    verbose=0
).flatten()


# ==========================================
# IMPROVED LSTM EVALUATION
# ==========================================

improved_mse, improved_rmse, improved_mae = calculate_metrics(
    y_test,
    improved_predictions
)


print("\nImproved LSTM Results")

print(
    "MSE :",
    improved_mse
)

print(
    "RMSE:",
    improved_rmse
)

print(
    "MAE :",
    improved_mae
)


# ==========================================
# PLOTS
# ==========================================

plot_training_history(
    improved_history
)

plot_predictions(
    y_test,
    improved_predictions,
    "Improved LSTM - Actual vs Predicted"
)


# ==========================================
# MODEL COMPARISON
# ==========================================

print("\n")
print("=" * 50)
print("MODEL COMPARISON")
print("=" * 50)


print(
    f"Basic LSTM RMSE    : {basic_rmse:.6f}"
)

print(
    f"Improved LSTM RMSE : {improved_rmse:.6f}"
)


if improved_rmse < basic_rmse:

    print(
        "\nImproved LSTM performed better."
    )

else:

    print(
        "\nBasic LSTM performed better."
    )


# ==========================================
# SAVE MODEL
# ==========================================

improved_model.save(
    "stock_price_lstm.keras"
)

print(
    "\nModel saved as stock_price_lstm.keras"
)

print(
    "\nPROJECT COMPLETED SUCCESSFULLY!"
)