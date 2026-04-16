"""
config.py — centralised settings loaded from .env via python-dotenv.

Usage:
    from config import settings
    print(settings.news_api_key)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (same directory as this file)
load_dotenv(Path(__file__).parent / ".env")

_ROOT = Path(__file__).parent


class Settings:
    # ------------------------------------------------------------------ #
    # NewsAPI                                                              #
    # ------------------------------------------------------------------ #
    news_api_key: str     = os.environ.get("NEWS_API_KEY", "")
    newsapi_base_url: str = "https://newsapi.org/v2"
    newsapi_page_size: int = 5       # articles per ticker (conserve free quota)
    newsapi_days: int     = 7        # look-back window in days

    # ------------------------------------------------------------------ #
    # FINRA short-interest API                                             #
    # ------------------------------------------------------------------ #
    finra_api_url: str = (
        "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
    )

    # ------------------------------------------------------------------ #
    # SEC EDGAR Form 4 RSS                                                 #
    # ------------------------------------------------------------------ #
    edgar_rss_url: str = "https://www.sec.gov/cgi-bin/browse-edgar"
    edgar_user_agent: str = os.environ.get(
        "EDGAR_USER_AGENT", "StockScreener research@example.com"
    )

    # ------------------------------------------------------------------ #
    # Database                                                             #
    # ------------------------------------------------------------------ #
    db_path: str = os.environ.get("DB_PATH", "./market.db")

    # ------------------------------------------------------------------ #
    # LLM backend                                                          #
    # ------------------------------------------------------------------ #
    llm_backend: str       = os.environ.get("LLM_BACKEND", "claude")
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    vllm_base_url: str     = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    llm_model_claude: str  = "claude-sonnet-4-20250514"
    llm_model_vllm: str    = os.environ.get("VLLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

    # ------------------------------------------------------------------ #
    # Cache TTLs (seconds)                                                 #
    # ------------------------------------------------------------------ #
    price_ttl: int     = 15 * 60     # 15 min  — OHLCV & fundamentals
    news_ttl: int      = 60 * 60     # 1 hr    — news articles
    iv_ttl: int        = 60 * 60     # 1 hr    — implied volatility
    earnings_ttl: int  = 6 * 60 * 60 # 6 hr   — earnings / SUE
    si_ttl: int        = 6 * 60 * 60 # 6 hr   — short interest
    insider_ttl: int   = 6 * 60 * 60 # 6 hr   — insider transactions

    # ------------------------------------------------------------------ #
    # Fetch concurrency & retry                                            #
    # ------------------------------------------------------------------ #
    semaphore_limit: int  = 30       # max concurrent yfinance / HTTP calls
    retry_attempts: int   = 3        # number of retry attempts
    retry_backoff_base: float = 2.0  # seconds; doubles on each retry

    # ------------------------------------------------------------------ #
    # Screener                                                             #
    # ------------------------------------------------------------------ #
    top_n: int = 20                  # number of top picks to return

    # ------------------------------------------------------------------ #
    # Model artefact paths                                                 #
    # ------------------------------------------------------------------ #
    models_dir: str = os.environ.get(
        "MODELS_DIR",
        str(_ROOT / "models"),
    )
    xgb_model_path: str            = str(Path(models_dir) / "xgb.json")
    feature_importance_path: str   = str(Path(models_dir) / "feature_importance.json")

    # ------------------------------------------------------------------ #
    # Fama-French                                                          #
    # ------------------------------------------------------------------ #
    ff_dataset: str = "F-F_Research_Data_5_Factors_2x3_daily"


settings = Settings()
