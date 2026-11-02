
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import torch

# Forex is inside the main Kronos project.
# Add the parent folder so Python can find the model package.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import Kronos, KronosTokenizer, KronosPredictor

selected_tokenizer = "NeoQuasar/Kronos-Tokenizer-base"
selected_model = "NeoQuasar/Kronos-base"
model_name = "Kronos-base"
MAX_CONTEXT = 512
PRED_LEN = 1

FORECAST_DIR = Path(__file__).resolve().parent / "forecasts"


# ==========================================
# 1. LOAD PRETRAINED KRONOS
# ==========================================


def load_kronos():

    import torch

    print(f"\nLoading {model_name} tokenizer...")

    tokenizer = KronosTokenizer.from_pretrained(
        selected_tokenizer
    )

    print(f"\nLoading pretrained {model_name}...")

    model = Kronos.from_pretrained(
        selected_model
    )

    torch.set_num_threads(4)

    predictor = KronosPredictor(
        model,
        tokenizer,
        max_context=512
    )

    print(f"\n{model_name} loaded successfully.")

    return predictor



# ==========================================
# 2. VALIDATE PREDICTED OHLC
# ==========================================

def validate_ohlc(candle):

    o = candle["open"]
    h = candle["high"]
    l = candle["low"]
    c = candle["close"]

    return (
        h >= max(o, c)
        and l <= min(o, c)
        and h >= l
    )


# ==========================================
# 3. SAVE PREDICTION
# ==========================================

def save_prediction(record, symbol):

    safe_symbol = symbol.replace("/", "_")

    folder = (
        FORECAST_DIR / safe_symbol / "1m"
    )

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    path = folder / "live_predictions.csv"

    new_row = pd.DataFrame([record])

    if path.exists():

        old_data = pd.read_csv(path)

        combined = pd.concat(
            [old_data, new_row],
            ignore_index=True
        )

        combined = combined.drop_duplicates(
            subset="target_timestamp",
            keep="last"
        )

    else:
        combined = new_row

    combined = combined.sort_values(
        "target_timestamp"
    )

    temp_path = path.with_name(
        "predictions_temp.csv"
    )

    combined.to_csv(
        temp_path,
        index=False
    )

    temp_path.replace(path)

    return path


# ==========================================
# 4. GENERATE NEXT CANDLE PREDICTION
# ==========================================

def predict_next_candle(
    predictor,
    history_df,
    temporary_candle,
    symbol
):

    if len(history_df) == 0:
        raise ValueError("Historical data is empty.")

    history = history_df.copy()

    history["timestamp"] = pd.to_datetime(
        history["timestamp"],
        utc=True
    )

    for col in ["open", "high", "low", "close"]:
        history[col] = pd.to_numeric(
            history[col],
            errors="coerce"
        )

    history = history.dropna(
        subset=[
            "timestamp",
            "open",
            "high",
            "low",
            "close"
        ]
    )

    history = history.sort_values(
        "timestamp"
    )

    history = history.drop_duplicates(
        subset="timestamp",
        keep="last"
    )

    # Prepare temporary candle.

    temp = dict(temporary_candle)

    temp["timestamp"] = pd.Timestamp(
        temp["timestamp"]
    )

    if temp["timestamp"].tzinfo is None:
        temp["timestamp"] = (
            temp["timestamp"].tz_localize("UTC")
        )
    else:
        temp["timestamp"] = (
            temp["timestamp"].tz_convert("UTC")
        )

    for col in ["open", "high", "low", "close"]:
        temp[col] = float(temp[col])

    if not validate_ohlc(temp):
        raise ValueError(
            "Temporary candle has invalid OHLC values."
        )

    if temp["timestamp"] <= history["timestamp"].iloc[-1]:
        raise ValueError(
            "Temporary candle must be newer than "
            "the latest completed candle."
        )

    # K completed candles + 1 temporary candle.
    # Maximum Kronos context = 2048.

    if len(history) + 1 > MAX_CONTEXT:
        history = history.tail(
            MAX_CONTEXT - 1
        )

    temp_df = pd.DataFrame([temp])

    model_data = pd.concat(
        [
            history[
                ["timestamp", "open", "high", "low", "close"]
            ],
            temp_df[
                ["timestamp", "open", "high", "low", "close"]
            ]
        ],
        ignore_index=True
    )

    x_df = model_data[
        ["open", "high", "low", "close"]
    ].astype(float)

    x_timestamp = pd.Series(
        pd.to_datetime(
            model_data["timestamp"],
            utc=True
        )
    )

    # Predict the candle immediately after the temporary candle.

    target_timestamp = (
        temp["timestamp"] + pd.Timedelta(minutes=1)
    )

    y_timestamp = pd.Series([
        target_timestamp
    ])

    print("\nRunning Kronos prediction...")

    print("Completed historical candles:", len(history))
    print("Temporary candles: 1")
    print("Total model context:", len(model_data))

    prediction = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=1,
        T=1.0,
        top_p=0.9,
        sample_count=1
    )

    if prediction is None or prediction.empty:
        raise RuntimeError(
            "Kronos returned an empty prediction."
        )

    first = prediction.iloc[0]

    predicted = {}

    for col in ["open", "high", "low", "close"]:

        if col not in prediction.columns:
            raise RuntimeError(
                f"Missing prediction column: {col}"
            )

        predicted[col] = float(first[col])

    # ======================================
    # 5. ALPHA AND DIRECTION
    # ======================================

    alpha = (
        predicted["close"] - predicted["open"]
    )

    pip_size = (
        0.01 if symbol.endswith("JPY")
        else 0.0001
    )

    expected_pips = alpha / pip_size

    if alpha > 0:
        direction = "GREEN"

    elif alpha < 0:
        direction = "RED"

    else:
        direction = "NEUTRAL"

    # ======================================
    # 6. OHLC VALIDATION
    # ======================================

    valid = validate_ohlc(predicted)

    # ======================================
    # 7. DISPLAY RESULT
    # ======================================

    snapshot_time = datetime.now(timezone.utc)

    print("\n========== FOREX PREDICTION ==========")

    print("Pair:", symbol)
    print("Snapshot:", snapshot_time)
    print("Target:", target_timestamp)

    print("Predicted Open :", predicted["open"])
    print("Predicted High :", predicted["high"])
    print("Predicted Low  :", predicted["low"])
    print("Predicted Close:", predicted["close"])

    print(f"Alpha: {alpha:+.6f}")
    print(f"Expected move: {expected_pips:+.2f} pips")

    print("Direction:", direction)

    if valid:
        print("OHLC validation: VALID")

    else:
        print(
            "WARNING: INVALID OHLC prediction. "
            "Raw values have not been changed."
        )

    print("======================================")

    # ======================================
    # 8. SAVE PREDICTION SEPARATELY
    # ======================================

    record = {
        "pair": symbol,
        "snapshot_time": snapshot_time.isoformat(),
        "target_timestamp": target_timestamp.isoformat(),
        "open": predicted["open"],
        "high": predicted["high"],
        "low": predicted["low"],
        "close": predicted["close"],
        "alpha": alpha,
        "direction": direction,
        "expected_pips": expected_pips,
        "ohlc_valid": valid
    }

    path = save_prediction(
        record,
        symbol
    )

    print("\nPrediction saved:", path)

    return record
