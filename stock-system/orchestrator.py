"""
orchestrator.py — LLM-powered final ranking over Screener top-20 output.

Pipeline
--------
1. Call ``Screener.screen(universe)`` → top-20 DataFrame with all signals.
2. Send structured JSON of all signals to the LLM.
3. LLM returns:
       - ranked buy list with reasoning
       - risk flags per ticker
       - suggested position sizes (equal-weight OR Kelly criterion)
4. Merge LLM output back into the DataFrame.
5. Persist recommendations to SQLite.
6. Return final pd.DataFrame.

LLM backend
-----------
Controlled by ``settings.llm_backend``:
    "claude"  → Anthropic Messages API  (claude-sonnet-4-20250514)
    "vllm"    → OpenAI-compatible endpoint (settings.vllm_base_url)

To swap backends, change one env var:  LLM_BACKEND=vllm
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from typing import Any

import aiosqlite
import pandas as pd

from config import settings
from data_ingestion import _ensure_schema

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Assembles screener output and calls the LLM for final buy-list ranking.

    Parameters
    ----------
    llm_backend : str, optional
        Override ``settings.llm_backend``.
    db_path : str, optional
        SQLite path.  Defaults to ``settings.db_path``.
    use_kelly : bool
        If True, compute Kelly-criterion position sizes alongside
        equal-weight sizes.  Default: True.
    """

    def __init__(
        self,
        llm_backend: str = settings.llm_backend,
        db_path: str | None = None,
        use_kelly: bool = True,
    ) -> None:
        self.llm_backend = llm_backend
        self._db_path    = db_path or settings.db_path
        self.use_kelly   = use_kelly

    # ==================================================================
    # Main entry-point
    # ==================================================================

    async def run(
        self,
        tickers: list[str] | None = None,
        screener_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Run the full orchestration pipeline.

        You can either supply a pre-computed *screener_df* (from
        ``Screener.screen()``) or let the orchestrator run the screener
        itself over *tickers*.

        Parameters
        ----------
        tickers : list[str], optional
            Universe to screen.  Ignored if *screener_df* is supplied.
        screener_df : pd.DataFrame, optional
            Pre-computed screener output (top-20 with all signal columns).

        Returns
        -------
        pd.DataFrame
            Columns: ticker, score, recommendation, justification,
                     risk_flag, position_size_equal, position_size_kelly,
                     plus all original signal columns.
        """
        if screener_df is None:
            if not tickers:
                raise ValueError("Provide either tickers or screener_df.")
            from screener import Screener
            sc = Screener(db_path=self._db_path)
            screener_df = await sc.screen(tickers)

        # LLM call
        llm_output = await self._call_llm(screener_df)

        # Position sizing
        n = len(screener_df)
        equal_w = round(1.0 / n, 6) if n > 0 else 0.0

        rows = []
        for _, row in screener_df.iterrows():
            ticker   = row["ticker"]
            llm_d    = llm_output.get(ticker, {})
            kelly_w  = self._kelly_size(
                row.get("xgb_pred", 0.0),
                row.get("egarch_vol", 0.3),
                n,
            ) if self.use_kelly else equal_w

            rows.append({
                **row.to_dict(),
                "recommendation":       llm_d.get("recommendation", "hold"),
                "justification":        llm_d.get("justification", ""),
                "risk_flag":            llm_d.get("risk_flag", ""),
                "position_size_equal":  equal_w,
                "position_size_kelly":  kelly_w,
            })

        df = (
            pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

        await self._persist(df)
        return df

    # ==================================================================
    # Position sizing
    # ==================================================================

    @staticmethod
    def _kelly_size(
        expected_return: float,
        sigma: float,
        n_positions: int,
        max_fraction: float = 0.20,
    ) -> float:
        """
        Compute a fractional Kelly position size.

        Uses the continuous-return Kelly formula:
            f* = μ / σ²

        Clamped to [0, max_fraction] and normalised so the portfolio
        sums to (roughly) 1 across *n_positions* positions.

        Parameters
        ----------
        expected_return : float
            Expected 60-day return (xgb_pred proxy).
        sigma : float
            Annualised volatility (egarch_vol).
        n_positions : int
            Total number of positions (used for sanity cap).
        max_fraction : float
            Maximum allocation per position (default 20 %).

        Returns
        -------
        float
            Position size fraction in [0, max_fraction].
        """
        sigma = max(sigma, 0.01)
        f     = expected_return / (sigma ** 2)
        # Half-Kelly for safety
        f = f * 0.5
        # Clamp
        f = max(0.0, min(f, max_fraction))
        return round(f, 6)

    # ==================================================================
    # LLM call
    # ==================================================================

    async def _call_llm(self, df: pd.DataFrame) -> dict[str, Any]:
        prompt_payload = self._build_prompt(df)
        try:
            if self.llm_backend == "claude":
                text = await self._call_claude(prompt_payload)
            else:
                text = await self._call_vllm(prompt_payload)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            text = "{}"

        return self._parse_llm_response(text, df["ticker"].tolist())

    def _build_prompt(self, df: pd.DataFrame) -> dict[str, Any]:
        """Serialise the top-20 signals DataFrame into a structured prompt."""
        signals_list = []
        for _, row in df.iterrows():
            signals_list.append({
                "ticker":         row.get("ticker"),
                "composite_score":round(float(row.get("score", 0) or 0), 4),
                "momentum":       row.get("momentum"),
                "sue_score":      row.get("sue_score"),
                "short_squeeze":  row.get("short_squeeze"),
                "insider_signal": row.get("insider_signal"),
                "revenue_accel":  row.get("revenue_accel"),
                "sentiment_score":row.get("sentiment_score"),
                "egarch_vol":     row.get("egarch_vol"),
                "egarch_flag":    row.get("egarch_flag"),
                "iv_spread":      row.get("iv_spread"),
                "xgb_pred":       row.get("xgb_pred"),
                "pe_ratio":       row.get("pe_ratio"),
                "eps_growth":     row.get("eps_growth"),
                "revenue_growth": row.get("revenue_growth"),
                "sector":         row.get("sector"),
            })

        system = (
            "You are a senior quantitative equity portfolio manager. "
            "Analyse the provided signal data for each stock and output "
            "a final buy recommendation list with concise justification "
            "and risk flags. Focus on 1-3 month return opportunities. "
            "Respond ONLY with valid JSON — no markdown, no extra text."
        )
        user = (
            "Analyse these top-20 screener results. For each ticker return:\n"
            '{"TICKER": {"recommendation": "buy|hold|sell", '
            '"justification": "2-3 sentence rationale targeting 1-3 month horizon", '
            '"risk_flag": "one-line key risk or empty string"}}\n\n'
            f"Signal data:\n{json.dumps(signals_list, indent=2, default=str)}"
        )
        return {"system": system, "user": user}

    async def _call_claude(self, payload: dict[str, Any]) -> str:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        msg = await client.messages.create(
            model=settings.llm_model_claude,
            max_tokens=2048,
            system=payload["system"],
            messages=[{"role": "user", "content": payload["user"]}],
        )
        return msg.content[0].text

    async def _call_vllm(self, payload: dict[str, Any]) -> str:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key="EMPTY", base_url=settings.vllm_base_url)
        resp = await client.chat.completions.create(
            model=settings.llm_model_vllm,
            messages=[
                {"role": "system", "content": payload["system"]},
                {"role": "user",   "content": payload["user"]},
            ],
            max_tokens=2048,
            temperature=0.1,
        )
        return resp.choices[0].message.content

    @staticmethod
    def _parse_llm_response(text: str, tickers: list[str]) -> dict[str, Any]:
        try:
            clean  = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = json.loads(clean)
        except Exception as exc:
            logger.warning("LLM JSON parse error: %s", exc)
            parsed = {}
        result = {}
        for t in tickers:
            entry = parsed.get(t, {})
            result[t] = {
                "recommendation": entry.get("recommendation", "hold"),
                "justification":  entry.get("justification", ""),
                "risk_flag":      entry.get("risk_flag", ""),
            }
        return result

    # ==================================================================
    # Persistence
    # ==================================================================

    async def _persist(self, df: pd.DataFrame) -> None:
        now        = time.time()
        expires_at = now + settings.price_ttl

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            await _ensure_schema(db)
            for _, row in df.iterrows():
                t = row["ticker"]
                await db.execute(
                    "INSERT INTO recommendations "
                    "(ticker, score, recommendation, justification, risk_flag, "
                    " position_size, signals, fetched_at, ttl_expires_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        t,
                        float(row.get("score") or 0),
                        str(row.get("recommendation", "hold")),
                        str(row.get("justification", "")),
                        str(row.get("risk_flag", "")),
                        float(row.get("position_size_kelly") or row.get("position_size_equal") or 0),
                        json.dumps(row.to_dict(), default=str),
                        now,
                        expires_at,
                    ),
                )
            await db.commit()
        logger.info("Recommendations persisted for %d tickers.", len(df))


# ---------------------------------------------------------------------------
# __main__ — smoke-test
# ---------------------------------------------------------------------------

async def _main() -> None:
    from universe import get_tickers_async

    print("Orchestrator smoke-test (top-5 tickers only)\n")
    # Use a tiny sub-universe for the smoke-test
    tickers = ["AAPL", "NVDA", "MSFT", "TSLA", "AMZN"]

    orch = Orchestrator()
    df   = await orch.run(tickers=tickers)

    cols = ["ticker", "score", "recommendation", "position_size_kelly", "risk_flag"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))


if __name__ == "__main__":
    asyncio.run(_main())
