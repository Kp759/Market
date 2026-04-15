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


class Settings:
    # ------------------------------------------------------------------ #
    # NewsAPI                                                              #
    # ------------------------------------------------------------------ #
    news_api_key: str    = os.environ.get("NEWS_API_KEY", "")
    newsapi_base_url: str = "https://newsapi.org/v2"
    newsapi_page_size: int = 10      # articles per ticker
    newsapi_days: int    = 7         # look-back window in days

    # ------------------------------------------------------------------ #
    # Database                                                             #
    # ------------------------------------------------------------------ #
    db_path: str = os.environ.get("DB_PATH", "./market.db")

    # ------------------------------------------------------------------ #
    # LLM backend                                                          #
    # ------------------------------------------------------------------ #
    llm_backend: str      = os.environ.get("LLM_BACKEND", "claude")  # "claude" | "vllm"
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    vllm_base_url: str    = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    llm_model_claude: str  = "claude-sonnet-4-20250514"
    llm_model_vllm: str   = os.environ.get("VLLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

    # ------------------------------------------------------------------ #
    # Cache TTLs (seconds)                                                 #
    # ------------------------------------------------------------------ #
    price_ttl: int = 15 * 60    # 15 minutes  — prices & fundamentals
    news_ttl: int  = 60 * 60    # 1 hour      — news articles
    iv_ttl: int    = 60 * 60    # 1 hour      — implied volatility

    # ------------------------------------------------------------------ #
    # Model artefact paths                                                 #
    # ------------------------------------------------------------------ #
    models_dir: str = os.environ.get(
        "MODELS_DIR",
        str(Path(__file__).parent / "models"),
    )
    xgb_model_path: str = str(Path(models_dir) / "xgb.json")

    # ------------------------------------------------------------------ #
    # Fama-French                                                          #
    # ------------------------------------------------------------------ #
    ff_dataset: str = "F-F_Research_Data_5_Factors_2x3_daily"


settings = Settings()
