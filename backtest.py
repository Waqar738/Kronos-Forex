
import time
from pathlib import Path

import pandas as pd

from config import SUPPORTED_PAIRS
from data_manager import update_dataset
from predictor import load_kronos


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_FOLDER = Path(__file__).resolve().parent

RESULTS_FOLDER = BASE_FOLDER / "backtests"

MAX_CONTEXT = 512

INITIAL_CONTEXT_PERCENT = 0.70


# --------------------------------------------------
# DIRECTION CLASSIFICATION
# --------------------------------------------------

def get_direction(open_price, close_price):

    if close_price > open_price:
        return "GREEN"

    elif close_price < open_price:
        return "RED"

    else:
        return "NEUTRAL"


# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------

def prepare_data(df, K):

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    for column in ["open", "high", "low", "close"]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=["timestamp", "open", "high", "low", "close"]
    )

    df = df.sort_values(
        "timestamp"
    )

    df = df.drop_duplicates(
        subset="timestamp",
        keep="last"
    )

    df = df.tail(K).reset_index(drop=True)

    if len(df) != K:

        raise ValueError(
            f"Required {K} candles, but only "
            f"{len(df)} valid candles are available."
        )

    # Validate actual OHLC data.
    invalid = (
        (df["high"] < df[["open", "close"]].max(axis=1))
        |
        (df["low"] > df[["open", "close"]].min(axis=1))
        |
        (df["high"] < df["low"])
    )

    if invalid.any():

        raise ValueError(
            f"Found {invalid.sum()} invalid historical candles. "
            "Fix the data before testing."
        )

    return df


# --------------------------------------------------
# PREDICT ONE CANDLE
# --------------------------------------------------

def predict_one_candle(predictor, history, target_timestamp):

    x_df = history[
        ["open", "high", "low", "close"]
    ].copy()

    x_timestamp = history["timestamp"]

    y_timestamp = pd.Series(
        [target_timestamp]
    )

    prediction = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=1,
        T=1.0,
        top_p=0.9,
        sample_count=1
    )

    if prediction is None or len(prediction) == 0:

        raise ValueError(
            "Kronos returned an empty prediction."
        )

    predicted = prediction.iloc[0]

    return {
        "pred_open": float(predicted["open"]),
        "pred_high": float(predicted["high"]),
        "pred_low": float(predicted["low"]),
        "pred_close": float(predicted["close"])
    }


# --------------------------------------------------
# RUN WALK-FORWARD BACKTEST
# --------------------------------------------------

def run_backtest(predictor, df, symbol, K):

    split_index = int(
        K * INITIAL_CONTEXT_PERCENT
    )

    initial_history = df.iloc[:split_index].copy()

    test_data = df.iloc[split_index:].copy()

    print("\n========== BACKTEST CONFIGURATION ==========")

    print(f"Pair: {symbol}")
    print(f"Total candles: {K}")
    print(f"Initial context: {len(initial_history)}")
    print(f"Test candles: {len(test_data)}")

    print("\nStarting walk-forward testing...")

    results = []

    start_time = time.perf_counter()

    for step in range(len(test_data)):

        actual = test_data.iloc[step]

        # Initial context plus previously revealed actual candles.
        history = df.iloc[:split_index + step].copy()

        target_timestamp = actual["timestamp"]

        print("\n" + "=" * 55)

        print(
            f"PREDICTION {step + 1} / {len(test_data)}"
        )

        print(f"Target time: {target_timestamp}")

        print(f"Input candles: {len(history)}")

        # Predict the next candle.
        predicted = predict_one_candle(
            predictor,
            history,
            target_timestamp
        )

        # Actual candle values.
        actual_open = float(actual["open"])
        actual_high = float(actual["high"])
        actual_low = float(actual["low"])
        actual_close = float(actual["close"])

        # Predicted direction.
        predicted_direction = get_direction(
            predicted["pred_open"],
            predicted["pred_close"]
        )

        # Actual direction.
        actual_direction = get_direction(
            actual_open,
            actual_close
        )

        # Direction comparison.
        correct = (
            predicted_direction == actual_direction
        )

        # Price errors.
        open_error = abs(
            predicted["pred_open"] - actual_open
        )

        high_error = abs(
            predicted["pred_high"] - actual_high
        )

        low_error = abs(
            predicted["pred_low"] - actual_low
        )

        close_error = abs(
            predicted["pred_close"] - actual_close
        )

        # Display prediction.
        print("\nPREDICTED")

        print(f"Open:  {predicted['pred_open']:.6f}")
        print(f"High:  {predicted['pred_high']:.6f}")
        print(f"Low:   {predicted['pred_low']:.6f}")
        print(f"Close: {predicted['pred_close']:.6f}")

        print(f"Direction: {predicted_direction}")

        # Display actual result.
        print("\nACTUAL")

        print(f"Open:  {actual_open:.6f}")
        print(f"High:  {actual_high:.6f}")
        print(f"Low:   {actual_low:.6f}")
        print(f"Close: {actual_close:.6f}")

        print(f"Direction: {actual_direction}")

        # Display result.
        print("\nRESULT")

        if correct:

            print("Direction: CORRECT")

        else:

            print("Direction: INCORRECT")

        print(f"Close price error: {close_error:.6f}")

        # Store result.
        result = {

            "timestamp": target_timestamp,

            "pred_open": predicted["pred_open"],
            "pred_high": predicted["pred_high"],
            "pred_low": predicted["pred_low"],
            "pred_close": predicted["pred_close"],

            "actual_open": actual_open,
            "actual_high": actual_high,
            "actual_low": actual_low,
            "actual_close": actual_close,

            "predicted_direction": predicted_direction,
            "actual_direction": actual_direction,

            "correct": correct,

            "open_error": open_error,
            "high_error": high_error,
            "low_error": low_error,
            "close_error": close_error
        }

        results.append(result)

        print("=" * 55)

        # The actual candle is now revealed.
        # It becomes part of the context for the next prediction.

    elapsed = time.perf_counter() - start_time

    results_df = pd.DataFrame(results)

    return results_df, elapsed


# --------------------------------------------------
# FINAL ACCURACY REPORT
# --------------------------------------------------


def calculate_streaks(results):

    longest_wrong = 0
    longest_correct = 0

    current_wrong = 0
    current_correct = 0

    for is_correct in results["correct"]:

        if is_correct:

            current_correct += 1
            current_wrong = 0

        else:

            current_wrong += 1
            current_correct = 0

        longest_wrong = max(
            longest_wrong,
            current_wrong
        )

        longest_correct = max(
            longest_correct,
            current_correct
        )

    return longest_wrong, longest_correct


def show_report(results, symbol, K, elapsed):

    total = len(results)

    if total == 0:

        print("No predictions were generated.")
        return

    green_predictions = (
        results["predicted_direction"] == "GREEN"
    ).sum()

    red_predictions = (
        results["predicted_direction"] == "RED"
    ).sum()

    neutral_predictions = (
        results["predicted_direction"] == "NEUTRAL"
    ).sum()

    green_correct = (
        (
            results["predicted_direction"] == "GREEN"
        )
        &
        (
            results["actual_direction"] == "GREEN"
        )
    ).sum()

    red_correct = (
        (
            results["predicted_direction"] == "RED"
        )
        &
        (
            results["actual_direction"] == "RED"
        )
    ).sum()

    neutral_correct = (
        (
            results["predicted_direction"] == "NEUTRAL"
        )
        &
        (
            results["actual_direction"] == "NEUTRAL"
        )
    ).sum()

    correct = int(
        results["correct"].sum()
    )

    incorrect = total - correct

    longest_wrong, longest_correct = calculate_streaks(results)

    accuracy = (
        correct / total
    ) * 100

    green_accuracy = (
        green_correct / green_predictions * 100
        if green_predictions else 0
    )

    red_accuracy = (
        red_correct / red_predictions * 100
        if red_predictions else 0
    )

    print("\n\n")
    print("=" * 55)

    print("          KRONOS BACKTEST FINAL REPORT")

    print("=" * 55)

    print(f"Currency pair: {symbol}")
    print(f"Historical candles: {K}")

    print(f"\nTotal predictions: {total}")

    print("\nOVERALL RESULTS")

    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Direction accuracy: {accuracy:.2f}%")

    print("\nCONSECUTIVE PERFORMANCE")

    print(
        f"Highest consecutive incorrect: {longest_wrong}"
    )

    print(
        f"Highest consecutive correct: {longest_correct}"
    )

    print("\nGREEN RESULTS")

    print(f"GREEN predictions: {green_predictions}")
    print(f"GREEN correct: {green_correct}")
    print(f"GREEN prediction accuracy: {green_accuracy:.2f}%")

    print("\nRED RESULTS")

    print(f"RED predictions: {red_predictions}")
    print(f"RED correct: {red_correct}")
    print(f"RED prediction accuracy: {red_accuracy:.2f}%")

    print("\nNEUTRAL RESULTS")

    print(f"NEUTRAL predictions: {neutral_predictions}")
    print(f"NEUTRAL correct: {neutral_correct}")

    print("\nAVERAGE PRICE ERROR")

    print(
        f"Open MAE:  {results['open_error'].mean():.8f}"
    )

    print(
        f"High MAE:  {results['high_error'].mean():.8f}"
    )

    print(
        f"Low MAE:   {results['low_error'].mean():.8f}"
    )

    print(
        f"Close MAE: {results['close_error'].mean():.8f}"
    )

    print("\nPERFORMANCE")

    print(f"Total time: {elapsed:.2f} seconds")

    print(f"Average time per prediction: {elapsed / total:.4f} seconds")

    print("=" * 55)


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():

    print("\n========== KRONOS WALK-FORWARD BACKTEST ==========")

    print("\nSupported currency pairs:")

    for key, pair in SUPPORTED_PAIRS.items():

        print(f"{key}. {pair}")

    choice = input("\nSelect pair (1, 2, or 3): ").strip()

    if choice not in SUPPORTED_PAIRS:

        print("Invalid selection.")
        return

    symbol = SUPPORTED_PAIRS[choice]

    try:

        K = int(
            input("\nEnter total historical candles (K): ")
        )

        if K < 10 or K > MAX_CONTEXT:

            print(
                f"K must be between 10 and {MAX_CONTEXT}."
            )

            return

    except ValueError:

        print("Enter a valid whole number.")
        return

    # ----------------------------------------------
    # DOWNLOAD / UPDATE DATA
    # ----------------------------------------------

    try:

        print("\nUpdating historical Forex data...")

        df = update_dataset(symbol, K)

        df = prepare_data(df, K)

        print(f"\nSuccessfully prepared {len(df)} candles.")

        print("\nInitial context:", int(K * 0.70))
        print("Testing candles:", K - int(K * 0.70))

    except Exception as error:

        print("\nDATA ERROR:")
        print(error)
        return

    # ----------------------------------------------
    # LOAD MODEL
    # ----------------------------------------------

    try:

        print("\nLoading pretrained Kronos model...")

        predictor = load_kronos()

        print("\nModel ready.")

    except Exception as error:

        print("\nMODEL ERROR:")
        print(error)
        return

    # ----------------------------------------------
    # RUN TEST
    # ----------------------------------------------

    try:

        results, elapsed = run_backtest(
            predictor,
            df,
            symbol,
            K
        )

        # ------------------------------------------
        # SAVE RESULTS
        # ------------------------------------------

        output_folder = (
            RESULTS_FOLDER
            / symbol.replace("/", "_")
            / "1m"
        )

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            output_folder / "backtest_results.csv"
        )

        results.to_csv(
            output_file,
            index=False
        )

        # ------------------------------------------
        # FINAL REPORT
        # ------------------------------------------

        show_report(
            results,
            symbol,
            K,
            elapsed
        )

        print("\nDetailed results saved:")
        print(output_file)

    except Exception as error:

        print("\nBACKTEST ERROR:")
        print(error)


if __name__ == "__main__":

    main()
