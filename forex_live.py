
from config import SUPPORTED_PAIRS
from data_manager import update_dataset, display_dataset

from predictor import load_kronos
from live_stream import ForexCandleEngine


def main():

    print("\n========== KRONOS FOREX SYSTEM ==========")

    # ------------------------------------------
    # STEP 1: SELECT CURRENCY PAIR
    # ------------------------------------------

    print("\nSupported currency pairs:")

    for key, pair in SUPPORTED_PAIRS.items():

        print(f"{key}. {pair}")

    choice = input("\nSelect pair (1, 2, or 3): ").strip()

    if choice not in SUPPORTED_PAIRS:

        print("Invalid selection.")
        return

    symbol = SUPPORTED_PAIRS[choice]

    # ------------------------------------------
    # STEP 2: ENTER HISTORICAL CANDLE COUNT
    # ------------------------------------------

    try:

        K = int(
            input("\nEnter historical candle count (K): ")
        )

        if K < 1 or K > 512:

            print(
                "K must be between 1 and 512."
            )

            return

    except ValueError:

        print("Please enter a valid number.")

        return

    print("\n========== SELECTED CONFIGURATION ==========")

    print(f"Currency pair: {symbol}")
    print(f"Historical candles: {K}")
    print("Temporary candle: 1")
    print(f"Total model context: {K + 1}")
    print("Timeframe: 1 minute")
    print("Prediction: Next 1-minute candle")

    # ------------------------------------------
    # STEP 3: LOAD KRONOS MODEL
    # ------------------------------------------

    try:

        print("\n========== LOADING KRONOS ==========")

        predictor = load_kronos()

        print("\nKronos model is ready.")

    except Exception as error:

        print("\nMODEL LOADING ERROR:")
        print(error)

        return

    # ------------------------------------------
    # STEP 4: UPDATE HISTORICAL DATA
    # ------------------------------------------

    try:

        print("\n========== UPDATING HISTORICAL DATA ==========")

        df = update_dataset(symbol, K)

        display_dataset(df, K)

        print("\nHistorical data setup completed.")

    except Exception as error:

        print("\nHISTORICAL DATA ERROR:")
        print(error)

        return

    # ------------------------------------------
    # STEP 5: START LIVE FOREX ENGINE
    # ------------------------------------------

    print("\n========== STARTING LIVE FOREX SYSTEM ==========")

    print("Kronos: Loaded")
    print("Historical data: Ready")
    print("Live feed: Twelve Data")
    print("Snapshot: At 55 seconds")
    print("Forecast storage: Separate from actual candles")

    engine = ForexCandleEngine(
        symbol=symbol,
        K=K,
        predictor=predictor
    )

    try:

        engine.start()

    except KeyboardInterrupt:

        print("\nForex system stopped by user.")

    except Exception as error:

        print("\nFOREX ENGINE ERROR:")
        print(error)


if __name__ == "__main__":

    main()
