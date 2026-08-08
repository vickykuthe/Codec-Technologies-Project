import pandas as pd
import yfinance as yf
from pathlib import Path


CSV_FILE = "google.csv"
TICKER = "GOOG"


def update_stock_data():

    print("=" * 60)
    print("GOOGLE STOCK DATA UPDATER")
    print("=" * 60)

    # Check existing CSV
    if not Path(CSV_FILE).exists():
        print(f"ERROR: {CSV_FILE} not found.")
        return

    old_data = pd.read_csv(CSV_FILE)

    print(f"\nExisting rows: {len(old_data)}")

    # Convert dates
    old_data["Date"] = pd.to_datetime(
        old_data["Date"],
        format="mixed",
        errors="coerce"
    )

    old_data = old_data.dropna(
        subset=["Date"]
    )

    # Find latest date
    latest_date = old_data["Date"].max()

    print(
        "Latest date in CSV:",
        latest_date.strftime("%Y-%m-%d")
    )

    # Download data after latest date
    print("\nDownloading latest GOOG data...")

    new_data = yf.download(
        TICKER,
        start=(latest_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        end=(pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False
    )

    if new_data.empty:

        print("\nNo new data available.")
        print("Your CSV is already up to date.")

        return

    # Handle yfinance column format
    if isinstance(new_data.columns, pd.MultiIndex):

        new_data.columns = [
            column[0]
            for column in new_data.columns
        ]

    new_data = new_data.reset_index()

    # Rename columns
    new_data = new_data.rename(
        columns={
            "Date": "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume"
        }
    )

    # Keep required columns
    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    new_data = new_data[
        required_columns
    ]

    # Convert dates
    new_data["Date"] = pd.to_datetime(
        new_data["Date"]
    )

    # Remove duplicate dates
    old_data = old_data[
        required_columns
    ]

    combined = pd.concat(
        [
            old_data,
            new_data
        ],
        ignore_index=True
    )

    combined = combined.drop_duplicates(
        subset=["Date"]
    )

    # Sort newest first
    combined = combined.sort_values(
        "Date",
        ascending=False
    ).reset_index(drop=True)

    # Save
    combined["Date"] = combined["Date"].dt.strftime(
        "%d-%b-%y"
    )

    combined.to_csv(
        CSV_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("DATA UPDATE SUCCESSFUL")
    print("=" * 60)

    print(
        f"Old rows: {len(old_data)}"
    )

    print(
        f"New rows downloaded: {len(new_data)}"
    )

    print(
        f"Total rows: {len(combined)}"
    )

    print(
        "Latest date:",
        combined.iloc[0]["Date"]
    )

    print(
        "\nUpdated file:",
        CSV_FILE
    )


if __name__ == "__main__":
    update_stock_data()
    