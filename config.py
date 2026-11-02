
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

if not API_KEY:
    raise ValueError(
        "Twelve Data API key not found. "
        "Add TWELVE_DATA_API_KEY to your .env file."
    )

BASE_URL = "https://api.twelvedata.com"

SUPPORTED_PAIRS = {
    "1": "EUR/USD",
    "2": "GBP/USD",
    "3": "USD/JPY",
}
