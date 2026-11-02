
import json
import time
import threading
import websocket
import pandas as pd

from pathlib import Path
from datetime import datetime, timezone

from config import API_KEY
from predictor import predict_next_candle


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

WS_URL = (
    "wss://ws.twelvedata.com/v1/quotes/price"
    f"?apikey={API_KEY}"
)

BASE_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = BASE_FOLDER / "market_data"

MAX_CONTEXT = 2048


# --------------------------------------------------
# FOREX CANDLE ENGINE
# --------------------------------------------------

class ForexCandleEngine:

    def __init__(self, symbol, K, predictor):

        self.symbol = symbol
        self.K = K
        self.predictor = predictor

        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.current_candle = None
        self.current_candle_complete = False

        self.latest_price = None

        self.last_snapshot_minute = None
        self.prediction_running = False

        self.ws = None
        self.price_count = 0

        folder = symbol.replace("/", "_")

        self.csv_path = (
            DATA_FOLDER / folder / "1m" / "candles.csv"
        )

    # --------------------------------------------------
    # TIME HELPERS
    # --------------------------------------------------

    def get_minute(self, timestamp):

        return timestamp.replace(
            second=0,
            microsecond=0
        )

    # --------------------------------------------------
    # CANDLE CREATION
    # --------------------------------------------------

    def create_candle(self, price, timestamp):

        return {
            "timestamp": timestamp,
            "open": price,
            "high": price,
            "low": price,
            "close": price
        }

    # --------------------------------------------------
    # UPDATE LIVE CANDLE
    # --------------------------------------------------

    def update_candle(self, price, received_time):

        minute = self.get_minute(received_time)

        completed_candle = None

        with self.lock:

            self.latest_price = price

            # First candle after program startup
            if self.current_candle is None:

                self.current_candle = self.create_candle(
                    price,
                    minute
                )

                # We joined in the middle of this minute.
                # Therefore, this candle is incomplete.
                self.current_candle_complete = False

                print(
                    f"\nStarted initial partial candle: {minute}"
                )

                return

            current_minute = self.current_candle["timestamp"]

            # A new minute has started
            if minute > current_minute:

                gap = int(
                    (minute - current_minute).total_seconds()
                    // 60
                )

                # Save the previous candle only when the
                # stream was active through a normal rollover.
                if self.current_candle_complete and gap == 1:

                    completed_candle = (
                        self.current_candle.copy()
                    )

                else:

                    print(
                        "\nWARNING: Previous candle was not "
                        "saved because its coverage was incomplete."
                    )

                if gap > 1:

                    print(
                        f"WARNING: {gap - 1} minute(s) "
                        "without received prices."
                    )

                # Start the new candle
                self.current_candle = self.create_candle(
                    price,
                    minute
                )

                # After a normal one-minute rollover,
                # the stream has been running continuously.
                self.current_candle_complete = (gap == 1)

                self.last_snapshot_minute = None

                print(
                    f"\nNew candle started: {minute}"
                )

            elif minute < current_minute:

                # Ignore older or out-of-order updates.
                return

            else:

                # Update current minute OHLC
                candle = self.current_candle

                candle["high"] = max(
                    candle["high"],
                    price
                )

                candle["low"] = min(
                    candle["low"],
                    price
                )

                candle["close"] = price

        # Save outside the lock so price reception is not blocked.
        if completed_candle is not None:

            self.save_completed_candle(completed_candle)

    # --------------------------------------------------
    # SAVE COMPLETED ACTUAL CANDLE
    # --------------------------------------------------

    def save_completed_candle(self, candle):

        if (
            candle["high"] < max(
                candle["open"],
                candle["close"]
            )
            or candle["low"] > min(
                candle["open"],
                candle["close"]
            )
            or candle["high"] < candle["low"]
        ):

            print(
                "\nERROR: Invalid OHLC candle. "
                "It was not saved."
            )

            return

        row = pd.DataFrame([candle])

        row["timestamp"] = pd.to_datetime(
            row["timestamp"],
            utc=True
        )

        for column in ["open", "high", "low", "close"]:

            row[column] = pd.to_numeric(
                row[column],
                errors="raise"
            )

        self.csv_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if self.csv_path.exists():

            existing = pd.read_csv(self.csv_path)

            if not existing.empty:

                existing["timestamp"] = pd.to_datetime(
                    existing["timestamp"],
                    utc=True
                )

                row = pd.concat(
                    [existing, row],
                    ignore_index=True
                )

        row = row.drop_duplicates(
            subset="timestamp",
            keep="last"
        )

        row = row.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        row.to_csv(
            self.csv_path,
            index=False
        )

        print("\nCOMPLETED ACTUAL CANDLE SAVED")
        print(f"Time: {candle['timestamp']}")
        print(f"Open: {candle['open']}")
        print(f"High: {candle['high']}")
        print(f"Low: {candle['low']}")
        print(f"Close: {candle['close']}")
        print(f"File: {self.csv_path}")

    # --------------------------------------------------
    # LOAD K COMPLETED HISTORICAL CANDLES
    # --------------------------------------------------

    def load_history(self, temporary_timestamp):

        if not self.csv_path.exists():

            raise FileNotFoundError(
                f"Historical CSV not found: {self.csv_path}"
            )

        history = pd.read_csv(self.csv_path)

        if history.empty:

            raise ValueError(
                "Historical candle CSV is empty."
            )

        history["timestamp"] = pd.to_datetime(
            history["timestamp"],
            utc=True,
            errors="coerce"
        )

        history = history.dropna(
            subset=["timestamp"]
        )

        for column in ["open", "high", "low", "close"]:

            history[column] = pd.to_numeric(
                history[column],
                errors="coerce"
            )

        history = history.dropna(
            subset=["open", "high", "low", "close"]
        )

        # Only candles earlier than the temporary candle.
        history = history[
            history["timestamp"] < temporary_timestamp
        ]

        history = history.drop_duplicates(
            subset="timestamp",
            keep="last"
        )

        history = history.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        if len(history) < self.K:

            raise ValueError(
                f"Need {self.K} completed historical candles, "
                f"but only {len(history)} are available."
            )

        # Select exactly K completed candles.
        history = history.tail(self.K).copy()

        return history

    # --------------------------------------------------
    # RUN KRONOS PREDICTION
    # --------------------------------------------------

    def prediction_worker(self, temporary_candle):

        start_time = time.perf_counter()

        try:

            temporary_timestamp = pd.to_datetime(
                temporary_candle["timestamp"],
                utc=True
            )

            print("\nLoading historical candles for Kronos...")

            history = self.load_history(
                temporary_timestamp
            )

            print(f"Historical candles: {len(history)}")
            print("Temporary candle: 1")
            print(
                f"Total model input: {len(history) + 1}"
            )

            print("\nStarting Kronos prediction...")

            # The predictor handles:
            # - Model input preparation
            # - One-candle forecast
            # - OHLC validation
            # - Alpha and direction
            # - Pip calculation
            # - Separate forecast CSV saving

            predict_next_candle(
                self.predictor,
                history,
                temporary_candle,
                self.symbol
            )

        except Exception as error:

            print("\nPREDICTION ERROR")
            print(error)

        finally:

            elapsed = time.perf_counter() - start_time

            print(
                f"\nTotal prediction processing time: "
                f"{elapsed:.2f} seconds"
            )

            if elapsed > 5:

                print(
                    "NOTICE: Prediction took more than "
                    "5 seconds."
                )

            with self.lock:

                self.prediction_running = False

    # --------------------------------------------------
    # 55-SECOND SNAPSHOT WORKER
    # --------------------------------------------------

    def snapshot_worker(self):

        while not self.stop_event.is_set():

            now = datetime.now(timezone.utc)

            if now.second >= 55:

                snapshot = None
                skip_reason = None

                with self.lock:

                    if self.current_candle is None:
                        time.sleep(0.1)
                        continue

                    current_minute = (
                        self.current_candle["timestamp"]
                    )

                    current_clock_minute = self.get_minute(now)

                    already_taken = (
                        self.last_snapshot_minute
                        == current_minute
                    )

                    if (
                        current_minute == current_clock_minute
                        and not already_taken
                        and self.latest_price is not None
                    ):

                        # Mark this minute as processed so
                        # another snapshot is not created.
                        self.last_snapshot_minute = (
                            current_minute
                        )

                        if not self.current_candle_complete:

                            skip_reason = (
                                "Current candle is incomplete "
                                "because of startup or a price gap."
                            )

                        elif self.prediction_running:

                            skip_reason = (
                                "Previous Kronos prediction is "
                                "still running."
                            )

                        else:

                            snapshot = (
                                self.current_candle.copy()
                            )

                            snapshot["close"] = self.latest_price

                            self.prediction_running = True

                if skip_reason is not None:

                    print("\n55-SECOND SNAPSHOT SKIPPED")
                    print(skip_reason)

                if snapshot is not None:

                    print("\n" + "=" * 50)

                    print("55-SECOND TEMPORARY CANDLE")
                    print(f"Pair: {self.symbol}")
                    print(f"Snapshot time: {now}")
                    print(
                        f"Temporary candle timestamp: "
                        f"{snapshot['timestamp']}"
                    )

                    print(f"Open:  {snapshot['open']}")
                    print(f"High:  {snapshot['high']}")
                    print(f"Low:   {snapshot['low']}")
                    print(f"Close: {snapshot['close']}")

                    print("Status: TEMPORARY - NOT SAVED")

                    print("=" * 50)

                    # Run Kronos in another thread.
                    # WebSocket price reception continues.
                    threading.Thread(
                        target=self.prediction_worker,
                        args=(snapshot,),
                        daemon=True
                    ).start()

            time.sleep(0.1)

    # --------------------------------------------------
    # WEBSOCKET EVENTS
    # --------------------------------------------------

    
    def on_open(self, ws):

        print("\nConnected to Twelve Data.")

        subscription = {
            "action": "subscribe",
            "params": {
                "symbols": self.symbol
            }
        }

        ws.send(json.dumps(subscription))

        print(f"Subscription requested: {self.symbol}")

    def on_message(self, ws, message):

        try:

            data = json.loads(message)

            event = data.get("event")

            if event == "subscribe-status":

                print("\nSubscription status:")
                print(data)

                if data.get("status") != "ok":

                    print("Subscription failed.")

                return

            if event == "price":

                price = float(data["price"])

                received_time = datetime.now(
                    timezone.utc
                )

                self.price_count += 1

                print(
                    f"{received_time.strftime('%H:%M:%S')} UTC"
                    f" | {self.symbol}"
                    f" | {price}"
                )

                self.update_candle(
                    price,
                    received_time
                )

                return

            print("Server message:", data)

        except Exception as error:

            print(
                "\nMessage processing error:",
                error
            )

            print("Raw message:", message)

    def on_error(self, ws, error):

        print("\nWebSocket error:", error)

    
    def on_close(self, ws, status, message):

        print("\nWebSocket connection closed.")
        print("Close status:", status)
        print("Close message:", message)

        if not self.stop_event.is_set():

            print("Connection lost. Reconnection will be attempted.")


    # --------------------------------------------------
    # START ENGINE
    # --------------------------------------------------

    def start(self):

        print("\n========== KRONOS FOREX CANDLE ENGINE ==========")
        print(f"Pair: {self.symbol}")
        print(f"Historical candles (K): {self.K}")
        print("Timeframe: 1 minute")
        print("Snapshot: 55 seconds")
        print("Automatic reconnection: Enabled")
        print("Waiting for live prices...\n")

        threading.Thread(
            target=self.snapshot_worker,
            daemon=True
        ).start()

        try:

            while not self.stop_event.is_set():

                print("\nConnecting to Twelve Data...")

                self.ws = websocket.WebSocketApp(
                    WS_URL,
                    on_open=self.on_open,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close
                )

                self.ws.run_forever(
                    ping_interval=20,
                    ping_timeout=10
                )

                if self.stop_event.is_set():
                    break

                print("\nRetrying connection in 5 seconds...")

                self.stop_event.wait(5)

        except KeyboardInterrupt:

            print("\nStopping Forex candle engine...")

        finally:

            self.stop_event.set()

            if self.ws:
                self.ws.close()

            print(
                f"\nTotal price updates: {self.price_count}"
            )

