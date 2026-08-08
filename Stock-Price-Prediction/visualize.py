import matplotlib.pyplot as plt


def plot_stock_price(data):
    """
    Plot historical stock closing price.
    """

    plt.figure(figsize=(12, 6))

    plt.plot(
        data["Date"],
        data["Close"],
        linewidth=1.5
    )

    plt.title("Google Stock Closing Price")

    plt.xlabel("Date")

    plt.ylabel("Closing Price")

    plt.xticks(rotation=45)

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.show()


def plot_predictions(actual, predicted, title):
    """
    Plot actual vs predicted values.
    """

    plt.figure(figsize=(12, 6))

    plt.plot(
        actual,
        label="Actual Price",
        linewidth=1.5
    )

    plt.plot(
        predicted,
        label="Predicted Price",
        linewidth=1.5
    )

    plt.title(title)

    plt.xlabel("Trading Days")

    plt.ylabel("Normalized Closing Price")

    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.show()


def plot_training_history(history):
    """
    Plot training and validation loss.
    """

    plt.figure(figsize=(10, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    if "val_loss" in history.history:

        plt.plot(
            history.history["val_loss"],
            label="Validation Loss"
        )

    plt.title("LSTM Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.show()