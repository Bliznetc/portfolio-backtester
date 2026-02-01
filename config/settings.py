# newsbot/config/settings.py
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

# Database Configuration (PostgreSQL)
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DATABASE", "stock_news_db"),
    "user": os.getenv("DB_USER", "your_user"),
    "port": int(os.getenv("DB_PORT", 5432)),  # Ensure port is an integer; Azure defaults to 5432
    "password": os.getenv("DB_PASSWORD", "your_password"),
}

# API Configuration
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "YOUR_FINNHUB_API_KEY")
# Rate limit for Finnhub client (requests per second)
FINNHUB_RATE_LIMIT_RPS = int(os.getenv("FINNHUB_RATE_LIMIT_RPS", 30))

# Which exchanges to index symbols for (comma-separated, defaults to US only)
_exchanges_env = os.getenv("FINNHUB_EXCHANGES", "US")
FINNHUB_EXCHANGES = [ex.strip() for ex in _exchanges_env.split(",") if ex.strip()]
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY") # e.g., from NewsAPI.org
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")

# Telegram Bot Admin ID (for sending direct alerts/logs)
TELEGRAM_ADMIN_ID = os.getenv("TELEGRAM_ADMIN_ID", None) # Optional, make sure it's an integer if used

# AI Model Configuration
SENTIMENT_MODEL_PATH = os.getenv("SENTIMENT_MODEL_PATH", "ProsusAI/finbert") # Example Hugging Face model
SENTIMENT_THRESHOLD_POSITIVE = float(os.getenv("SENTIMENT_THRESHOLD_POSITIVE", 0.6))
SENTIMENT_THRESHOLD_NEGATIVE = float(os.getenv("SENTIMENT_THRESHOLD_NEGATIVE", 0.4)) # Lower than positive threshold

# News Retrieval Configuration
NEWS_LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", 1)) # How many days back to fetch news