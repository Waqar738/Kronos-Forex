# Kronos Forex Prediction System

A Python-based Forex market analysis and prediction system powered by the pretrained Kronos time-series foundation model.

The project connects to Twelve Data for historical and live Forex market data. It uses historical OHLC candles as model input, generates next-candle predictions, and evaluates model performance through historical walk-forward backtesting.

## 1. Main Features

- Historical Forex candle downloading.
- Support for EUR/USD, GBP/USD, and USD/JPY.
- One-minute candle data.
- Live price streaming through WebSocket.
- Automatic live candle construction.
- Next-candle OHLC prediction using Kronos.
- GREEN, RED, and NEUTRAL market direction classification.
- Automatic prediction storage.
- Historical walk-forward backtesting.
- Consecutive correct and incorrect prediction analysis.
- Separate storage for market data, live forecasts, and backtesting results.

**Important:** This is a research and decision-support project. Predictions are model outputs, not guaranteed market outcomes or financial advice.

## 2. Project Structure

```text
Forex/
│
├── config.py
├── data_manager.py
├── forex_live.py
├── live_stream.py
├── predictor.py
├── backtest.py
├── test_model.py
│
├── market_data/
│   └── EUR_USD/
│       └── 1m/
│           └── candles.csv
│
├── forecasts/
│   └── EUR_USD/
│       └── 1m/
│           └── live_predictions.csv
│
├── backtests/
│   └── EUR_USD/
│       └── 1m/
│           └── backtest_results.csv
│
├── .env
└── README.md
```

### Python Files

| File | Purpose |
|---|---|
| `config.py` | Stores API configuration and supported currency pairs. |
| `data_manager.py` | Downloads, updates, validates, and stores historical candle data. |
| `test_model.py` | Tests whether the Kronos model and tokenizer can load successfully. |
| `predictor.py` | Loads the pretrained Kronos model, generates predictions, validates OHLC values, and saves forecasts. |
| `live_stream.py` | Receives live price updates, constructs candles, manages WebSocket connections, and triggers predictions. |
| `forex_live.py` | Main program for selecting a currency pair, downloading historical data, and starting live prediction. |
| `backtest.py` | Evaluates the model using historical walk-forward testing. |

## 3. Requirements

- Python 3.11
- Windows 10 or Windows 11
- Internet connection
- Twelve Data API key
- Git
- PyTorch
- Kronos pretrained model and tokenizer

The system can run using CPU inference. A GPU is not required.

## 4. Installation

### Step 1: Clone the repository

```cmd
git clone https://github.com/YOUR_USERNAME/Kronos-Forex.git
```

Move into the project folder:

```cmd
cd Kronos-Forex
```

### Step 2: Create a virtual environment

```cmd
python -m venv kronos311_env
```

Activate it:

```cmd
kronos311_env\Scripts\activate
```

### Step 3: Install dependencies

```cmd
python -m pip install --upgrade pip
```

Install the required packages used by the project:

```cmd
pip install pandas numpy requests websocket-client python-dotenv torch
```

The Kronos model code must also be available. Clone the original Kronos repository alongside the Forex folder if it is not already available:

```cmd
git clone https://github.com/shiyu-coder/Kronos.git Kronos
```

Make sure the `model` package is accessible to the Forex scripts through their configured Python import path.

## 5. API Configuration

The project uses Twelve Data for historical and live Forex market data.

Create a `.env` file inside the Forex folder:

```env
TWELVE_DATA_API_KEY=YOUR_API_KEY_HERE
```

Replace `YOUR_API_KEY_HERE` with your own API key.

Never upload your `.env` file or expose your API key publicly.

## 6. Kronos Model Configuration

The project uses pretrained Kronos models. No model training or fine-tuning is required for normal prediction.

The model and tokenizer must be compatible.

Examples of supported model configurations:

**Kronos-mini**

```python
tokenizer = KronosTokenizer.from_pretrained(
    "NeoQuasar/Kronos-Tokenizer-2k"
)

model = Kronos.from_pretrained(
    "NeoQuasar/Kronos-mini"
)
```

Maximum context: 2048.

**Kronos-small**

```python
tokenizer = KronosTokenizer.from_pretrained(
    "NeoQuasar/Kronos-Tokenizer-base"
)

model = Kronos.from_pretrained(
    "NeoQuasar/Kronos-small"
)
```

Maximum context: 512.

**Kronos-base**

```python
tokenizer = KronosTokenizer.from_pretrained(
    "NeoQuasar/Kronos-Tokenizer-base"
)

model = Kronos.from_pretrained(
    "NeoQuasar/Kronos-base"
)
```

Maximum context: 512.

The selected model configuration must match the `MAX_CONTEXT` setting in the prediction program.

The pretrained model is downloaded automatically when using `from_pretrained()` for the first time.

## 7. Test the Model

Before starting the Forex system, check whether the model and tokenizer load successfully.

Run:

```cmd
python test_model.py
```

A successful test should confirm that the tokenizer and selected Kronos model have loaded.

## 8. Historical Data Management

The `data_manager.py` program handles historical candle data.

The system supports these currency pairs:

- EUR/USD
- GBP/USD
- USD/JPY

The historical data manager:

1. Connects to Twelve Data.
2. Requests historical one-minute candles.
3. Preserves previously downloaded candles.
4. Downloads missing recent candles.
5. Removes duplicate timestamps.
6. Sorts candles chronologically.
7. Checks for missing timestamp intervals.
8. Saves the updated dataset as CSV.

Historical candle data contains:

- Timestamp
- Open
- High
- Low
- Close

The current forming candle is excluded from the completed historical dataset.

## 9. Live Forex Prediction

The main live program is:

```cmd
python forex_live.py
```

### How to use it

**Step 1: Select a currency pair**

The program displays:

```text
1. EUR/USD
2. GBP/USD
3. USD/JPY
```

Enter your selection.

**Step 2: Enter historical candle count**

Enter the number of completed historical candles (K) that you want to use as model context.

The maximum depends on the selected Kronos model:

- Kronos-mini: 2048
- Kronos-small: 512
- Kronos-base: 512

**Step 3: Historical data preparation**

The program updates the historical dataset and displays the selected candles.

**Step 4: Load the model**

The pretrained tokenizer and Kronos model are loaded.

**Step 5: Start live streaming**

The system connects to the Twelve Data WebSocket and receives live price updates.

**Step 6: Build live candles**

The system collects incoming prices and constructs one-minute OHLC candles.

**Step 7: Generate prediction**

Near the end of the current minute, the system creates a temporary candle snapshot and uses it with completed historical candles to generate the next-candle prediction.

The prediction is generated without waiting for the current candle to finish.

**Step 8: Display the result**

The system displays:

- Currency pair
- Snapshot timestamp
- Target timestamp
- Predicted Open
- Predicted High
- Predicted Low
- Predicted Close
- Alpha
- Expected movement in pips
- Direction
- OHLC validation
- Processing time

### Direction Classification

The direction is calculated using predicted Open and Close:

- GREEN: Predicted Close is greater than Predicted Open.
- RED: Predicted Close is lower than Predicted Open.
- NEUTRAL: Predicted Close equals Predicted Open.

The predicted OHLC values are also checked for valid candle structure.

### Live Prediction Storage

Predictions are saved separately from actual historical market data:

```text
forecasts/
└── EUR_USD/
    └── 1m/
        └── live_predictions.csv
```

This keeps predicted values separate from actual market candles.

## 10. Historical Walk-Forward Backtesting

The backtesting program is:

```cmd
python backtest.py
```

The purpose is to evaluate the pretrained model using historical data that is revealed one candle at a time.

No model training or fine-tuning takes place.

### Backtesting Process

Suppose you select 1,000 historical candles.

**Step 1: Divide the dataset**

- First 70%: Initial historical input context.
- Remaining 30%: Testing candles.

For 1,000 candles:

- 700 candles form the initial context.
- 300 candles are tested one by one.

**Step 2: Generate the first prediction**

The model receives the initial historical context and predicts the next candle.

**Step 3: Compare with actual data**

The predicted candle is compared with the actual historical candle.

The program checks:

- Predicted direction.
- Actual direction.
- Correct or incorrect direction.
- Predicted Open versus actual Open.
- Predicted High versus actual High.
- Predicted Low versus actual Low.
- Predicted Close versus actual Close.

**Step 4: Update historical context**

After evaluating the prediction, the actual candle is added to the input history.

**Step 5: Generate the next prediction**

The model uses the updated historical context to predict the following candle.

This process continues until all testing candles have been evaluated.

### Important Backtesting Rule

The actual testing candle is not included in the model input before its prediction.

It is revealed only after the prediction has been generated.

This prevents future-candle information from leaking into the prediction process.

## 11. Backtesting Performance Report

The backtesting program reports:

- Total predictions.
- Correct predictions.
- Incorrect predictions.
- Overall direction accuracy.
- Total GREEN predictions.
- Total RED predictions.
- Total NEUTRAL predictions.
- Correct GREEN predictions.
- Correct RED predictions.
- GREEN and RED prediction accuracy.
- Average Open price error.
- Average High price error.
- Average Low price error.
- Average Close price error.
- Total processing time.

### Consecutive Prediction Analysis

The system also calculates:

- Highest consecutive incorrect predictions.
- Highest consecutive correct predictions.

For example:

```text
Prediction results:

Correct
Incorrect
Incorrect
Incorrect
Incorrect
Correct
Incorrect
Incorrect
```

Highest consecutive incorrect predictions: 4.

A correct prediction resets the incorrect streak.

## 12. Backtesting Results Storage

Backtesting results are stored separately:

```text
backtests/
└── EUR_USD/
    └── 1m/
        └── backtest_results.csv
```

This keeps historical evaluation results separate from:

- Actual market data.
- Live forecasts.

## 13. Important Notes

1. The project uses pretrained Kronos inference, not custom model training.
2. Historical data and live forecasts are stored separately.
3. Predictions are based on available historical and live candle information.
4. Model accuracy does not guarantee profitable trading.
5. Backtesting direction accuracy does not include spread, transaction costs, slippage, or execution conditions.
6. The maximum input context depends on the selected Kronos model.
7. A larger model does not automatically guarantee better Forex prediction performance.
8. Historical data gaps can affect evaluation quality.

## 14. Troubleshooting

**API connection error**

- Check your internet connection.
- Verify your Twelve Data API key.
- Check API usage limits.

**Model loading error**

- Check that the Kronos repository is accessible.
- Verify the model and tokenizer names.
- Check your internet connection for the initial download.

**Context-length error**

- Reduce K.
- Check that `MAX_CONTEXT` matches the selected model.

**WebSocket disconnection**

- Check internet connectivity.
- The live stream uses reconnection handling.
- Check the displayed connection messages.

**Missing Python package**

Install the missing package using:

```cmd
pip install PACKAGE_NAME
```

## 15. Project Goal

The goal of Kronos Forex is to explore the practical use of pretrained time-series foundation models in financial market analysis.

It combines historical market data, live price streaming, next-candle forecasting, and walk-forward evaluation in one Python-based system.

The project is intended for research, experimentation, and learning about AI-based financial time-series analysis.

---

**Developer:** Muhammad Waqar

**Model:** Kronos Time-Series Foundation Model

**Market Data Provider:** Twelve Data

**Programming Language:** Python

**Project Type:** AI-Based Forex Prediction and Historical Backtesting

**GitHub:** https://github.com/Waqar738
