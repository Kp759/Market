"""
orchestrator.py — LLM-powered orchestrator that synthesises price data,
                  fundamentals, sentiment, and forecasts into a coherent
                  investment brief.

LLM backend abstraction
-----------------------
Set LLM_BACKEND in .env (or config.py) to switch between backends:

    "claude"  →  Anthropic Claude API  (default)
    "vllm"    →  self-hosted vLLM endpoint (OpenAI-compatible)

Only this file needs to change when swapping backends.

Status: stub — implementation coming in the next sprint.
"""

from __future__ import annotations

from typing import Any

from config import settings


class Orchestrator:
    """
    LLM orchestrator.

    Parameters
    ----------
    llm_backend : str
        Override the backend from settings.  Accepts "claude" or "vllm".
    """

    def __init__(self, llm_backend: str = settings.llm_backend) -> None:
        self.llm_backend = llm_backend

    async def analyse(self, stock_data: dict[str, Any]) -> dict[str, Any]:
        """
        Synthesise all signals into a per-ticker investment brief.

        Parameters
        ----------
        stock_data : dict
            Output of DataIngestion.get_stock_data().

        Returns
        -------
        dict
            Keyed by ticker; value is an LLM-generated brief + raw signals.
        """
        raise NotImplementedError(
            f"Orchestrator (backend={self.llm_backend!r}) is not yet implemented."
        )

    # ------------------------------------------------------------------
    # Private helpers (to be implemented)
    # ------------------------------------------------------------------

    async def _call_claude(self, prompt: str) -> str:
        """Call Anthropic Claude API."""
        raise NotImplementedError

    async def _call_vllm(self, prompt: str) -> str:
        """Call self-hosted vLLM OpenAI-compatible endpoint."""
        raise NotImplementedError
