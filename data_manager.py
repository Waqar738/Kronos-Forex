
import requests
import pandas as pd

from pathlib import Path
from datetime import datetime, timezone, timedelta

from config import API_KEY, BASE_URL


DATA_FOLDER = Path("market_data")
INTERVAL = "1min"
MAX_CANDLES_PER_REQUEST = 5000

COLUMNS = ["timestamp", "open", "high", "low", "close"]


def get_dataset_path(symbol):
    folder_name = symbol.replace("/", "_")
    return DATA_FOLDER / folder_name / "1m" / "candles.csv"


def fetch_candles(symbol, outputsize=100, start_date=None):
    """
    Download 1-minute Forex candles from Twelve Data.

    All timestamps are requested in UTC.
    """

    url = f"{BASE_URL}/time_series"

    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "outputsize": min(outputsize, MAX_CANDLES_PER_REQUEST),
        "order": "ASC",
        "timezone": "UTC",
        "apikey": API_KEY,
        "format": "JSON",
    }

    if start_date is not None:
        params["start_date"] = start_date.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    print(f"\nDownloading {symbol} candles...")

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    result = response.json()

    if result.get("status") == "error" or "values" not in result:
        raise RuntimeError(
            f"Twelve Data error: {result.get('message', result)}"
        )

    df = pd.DataFrame(result["values"])

    if df.empty:
        return pd.DataFrame(columns=COLUMNS)

    df = df.rename(columns={"datetime": "timestamp"})

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True
    )

    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df[COLUMNS]

    df = df.dropna(subset=COLUMNS)

    # Only accept completed candles.
    current_minute = pd.Timestamp.now(
        tz="UTC"
    ).floor("min")

    df = df[df["timestamp"] < current_minute]

    df = df.sort_values("timestamp")
    df = df.drop_duplicates(
        subset="timestamp", keep="last"
    )

    return df.reset_index(drop=True)


def load_existing_data(path):
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)

    df = pd.read_csv(path)

    if df.empty:
        return pd.DataFrame(columns=COLUMNS)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True
    )

    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df[COLUMNS]
    df = df.dropna(subset=COLUMNS)

    return df.sort_values("timestamp").reset_index(drop=True)


def save_dataset(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    df = df.sort_values("timestamp")
    df = df.drop_duplicates(
        subset="timestamp", keep="last"
    )

    df.to_csv(path, index=False)

    print(f"Dataset saved: {path}")
    print(f"Total candles: {len(df)}")


def update_dataset(symbol, K):
    """
    Preserve existing candles and append missing recent candles.

    If history is insufficient, download recent history to
    increase the dataset to at least K candles.
    """

    path = get_dataset_path(symbol)

    existing = load_existing_data(path)

    print(f"\nExisting candles: {len(existing)}")

    if existing.empty:
        print("No local dataset found. Creating new dataset.")

        new_data = fetch_candles(
            symbol,
            outputsize=K + 10
        )

        combined = new_data

    else:
        latest_timestamp = existing["timestamp"].max()

        next_timestamp = (
            latest_timestamp + timedelta(minutes=1)
        )

        print(
            f"Latest local candle: {latest_timestamp}"
        )

        # Download candles newer than the latest local candle.
        new_data = fetch_candles(
            symbol,
            outputsize=MAX_CANDLES_PER_REQUEST,
            start_date=next_timestamp
        )

        combined = pd.concat(
            [existing, new_data],
            ignore_index=True
        )

        # If the local dataset has fewer than K candles,
        # retrieve recent history to fill the missing amount.
        if len(combined) < K:
            print(
                "\nNot enough local history. "
                "Downloading additional recent candles."
            )

            additional_data = fetch_candles(
                symbol,
                outputsize=K + 10
            )

            combined = pd.concat(
                [combined, additional_data],
                ignore_index=True
            )

    combined = combined.drop_duplicates(
        subset="timestamp", keep="last"
    )

    combined = combined.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    save_dataset(combined, path)

    if len(combined) < K:
        print(
            f"\nWARNING: Requested {K} candles, "
            f"but only {len(combined)} are available."
        )

    return combined


def display_dataset(df, K):
    print("\n========== DATASET SUMMARY ==========")

    print(f"Available candles: {len(df)}")
    print(f"Required candles: {K}")

    if not df.empty:
        print(f"First candle: {df.iloc[0]['timestamp']}")
        print(f"Latest candle: {df.iloc[-1]['timestamp']}")

        print("\nLast 5 candles:")
        print(df.tail(5).to_string(index=False))

    print("=====================================")
