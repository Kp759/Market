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
    news_api_key: str = os.environ.get("NEWS_API_KEY", "")

    # ------------------------------------------------------------------ #
    # Database                                                             #
    # ------------------------------------------------------------------ #
    db_path: str = os.environ.get("DB_PATH", "./market.db")

    # ------------------------------------------------------------------ #
    # LLM backend                                                          #
    # ------------------------------------------------------------------ #
    llm_backend: str = os.environ.get("LLM_BACKEND", "claude")  # "claude" | "vllm"
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    vllm_base_url: str = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")

    # ------------------------------------------------------------------ #
    # Cache TTLs (seconds)                                                 #
    # ------------------------------------------------------------------ #
    price_ttl: int = 15 * 60       # 15 minutes
    news_ttl: int  = 60 * 60       # 1 hour

    # ------------------------------------------------------------------ #
    # NewsAPI                                                              #
    # ------------------------------------------------------------------ #
    newsapi_base_url: str = "https://newsapi.org/v2"
    newsapi_page_size: int = 5     # articles per ticker (conserve free quota)


settings = Settings()
