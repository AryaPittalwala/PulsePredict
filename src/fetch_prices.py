"""
fetch_prices.py – Batched, retried, incremental yfinance downloader.

Usage (as a module entry-point):
    python -m src.fetch_prices [--force-full] [--universe nifty50|nifty100]

Design decisions
----------------
* Downloads in batches of DOWNLOAD_BATCH_SIZE tickers with exponential
  backoff and a sleep between batches to avoid rate-limiting.
* Timestamps are normalised to Asia/Kolkata and filtered to market hours.
* Missing bars within a session are left missing (never forward-filled across
  days) — this is intentional per the spec.
* Each run upserts only the last FETCH_DAYS of data so the SQLite DB grows
  incrementally.  Pass --force-full to fetch all 60 days.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import pytz
import yfinance as yf

# Allow running as `python -m src.fetch_prices` from the project root
sys.path.insert(0, ".")

from config import (
    DOWNLOAD_BATCH_SIZE,
    DOWNLOAD_RETRIES,
    DOWNLOAD_SLEEP_SEC,
    FETCH_DAYS,
    HISTORY_DAYS,
    INDEX_TICKER,
    INTERVAL,
    MARKET_CLOSE,
    MARKET_OPEN,
    TZ,
)
from src.db import init_db, upsert_prices
from src.universe import get_nifty100, get_nifty50

logger = logging.getLogger(__name__)

IST = pytz.timezone(TZ)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_ist(ts: pd.DatetimeTZDtype | pd.Timestamp) -> pd.Timestamp:
    """Convert a timezone-aware timestamp to IST."""
    if ts.tzinfo is None:
        # Treat naive timestamps as UTC (yfinance sometimes returns them)
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(TZ)


def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows within NSE market hours (09:15 – 15:30 IST)."""
    if df.empty:
        return df
    times = df.index.time
    open_t = datetime.strptime(MARKET_OPEN, "%H:%M").time()
    close_t = datetime.strptime(MARKET_CLOSE, "%H:%M").time()
    mask = (times >= open_t) & (times <= close_t)
    return df[mask]


def _download_batch(
    tickers: list[str],
    period: str = "5d",
    retries: int = DOWNLOAD_RETRIES,
) -> dict[str, pd.DataFrame]:
    """
    Download 5-minute OHLCV for a batch of tickers with retry/backoff.

    Returns a dict {ticker: DataFrame} with IST-indexed rows.
    """
    result: dict[str, pd.DataFrame] = {}
    for attempt in range(retries):
        try:
            raw = yf.download(
                tickers,
                period=period,
                interval=INTERVAL,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                logger.warning("Empty response for batch %s (attempt %d)", tickers, attempt + 1)
                break

            for ticker in tickers:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        if ticker in raw.columns.get_level_values(0):
                            df = raw[ticker].copy()
                        elif ticker in raw.columns.get_level_values(1):
                            df = raw.xs(ticker, level=1, axis=1).copy()
                        else:
                            logger.warning("Ticker %s missing from batch response.", ticker)
                            continue
                    else:
                        df = raw.copy()

                    if df.empty:
                        continue

                    # Normalise index to IST
                    if df.index.tzinfo is None:
                        df.index = df.index.tz_localize("UTC")
                    df.index = df.index.tz_convert(TZ)

                    # Filter market hours
                    df = _filter_market_hours(df)
                    if df.empty:
                        continue

                    # Rename columns (auto_adjust gives Open High Low Close Volume)
                    df.columns = [c.lower() for c in df.columns]
                    df = df[["open", "high", "low", "close", "volume"]]
                    df.index.name = "ts"
                    result[ticker] = df

                except Exception as e:
                    logger.error("Error processing ticker %s: %s", ticker, e)

            return result  # success

        except Exception as e:
            wait = 2 ** attempt * DOWNLOAD_SLEEP_SEC
            logger.warning(
                "Batch download attempt %d/%d failed (%s). Retrying in %.1fs…",
                attempt + 1, retries, e, wait,
            )
            time.sleep(wait)

    logger.error("All %d attempts failed for batch %s.", retries, tickers)
    return result


def _df_to_rows(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """Convert an OHLCV DataFrame to the flat format expected by upsert_prices."""
    out = df.copy()
    out["ticker"] = ticker
    out["ts"] = out.index.strftime("%Y-%m-%d %H:%M:%S%z")
    return out.reset_index(drop=True)[["ticker", "ts", "open", "high", "low", "close", "volume"]]


# ── Main download logic ────────────────────────────────────────────────────────

def fetch_and_store(
    tickers: list[str],
    days_back: int = FETCH_DAYS,
) -> dict[str, int]:
    """
    Fetch the last ``days_back`` days of 5-minute data and upsert to SQLite.

    Returns a dict {ticker: row_count_stored}.
    """
    end_dt = datetime.now(IST)
    start_dt = end_dt - timedelta(days=days_back + 2)  # +2 to absorb weekends

    period_str = f"{min(days_back, 60)}d"
    stored: dict[str, int] = {}

    for i in range(0, len(tickers), DOWNLOAD_BATCH_SIZE):
        batch = tickers[i : i + DOWNLOAD_BATCH_SIZE]
        logger.info(
            "Downloading batch %d/%d: %s",
            i // DOWNLOAD_BATCH_SIZE + 1,
            (len(tickers) + DOWNLOAD_BATCH_SIZE - 1) // DOWNLOAD_BATCH_SIZE,
            batch,
        )
        batch_data = _download_batch(batch, period=period_str)

        for ticker, df in batch_data.items():
            flat = _df_to_rows(ticker, df)
            n = upsert_prices(flat)
            stored[ticker] = n
            logger.info("  %s: stored %d rows", ticker, n)

        if i + DOWNLOAD_BATCH_SIZE < len(tickers):
            time.sleep(DOWNLOAD_SLEEP_SEC)

    return stored


def fetch_full_history(tickers: list[str]) -> dict[str, int]:
    """Fetch the full ~60-day history available from yfinance."""
    return fetch_and_store(tickers, days_back=HISTORY_DAYS)


def fetch_index(days_back: int = FETCH_DAYS) -> int:
    """Fetch and store the NIFTY 50 index bars."""
    result = fetch_and_store([INDEX_TICKER], days_back=days_back)
    return result.get(INDEX_TICKER, 0)


def print_summary(tickers: list[str]) -> None:
    """Print a per-ticker row count and coverage summary."""
    from src.db import get_ticker_date_range, get_ticker_row_count

    print(f"\n{'Ticker':<20} {'Rows':>8} {'Min TS':<26} {'Max TS':<26}")
    print("-" * 82)
    ok = 0
    for t in sorted(tickers):
        n = get_ticker_row_count(t)
        mn, mx = get_ticker_date_range(t)
        status = "✓" if n > 40 * 75 * 0.5 else "✗"  # rough check: >50% of expected
        print(f"{t:<20} {n:>8}  {str(mn):<26} {str(mx):<26} {status}")
        if n > 0:
            ok += 1
    pct = 100 * ok / len(tickers) if tickers else 0
    print(f"\nTotal: {ok}/{len(tickers)} tickers have data ({pct:.1f}%)")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Fetch NSE intraday price data.")
    parser.add_argument(
        "--universe",
        choices=["nifty50", "nifty100"],
        default="nifty100",
        help="Ticker universe to download (default: nifty100)",
    )
    parser.add_argument(
        "--force-full",
        action="store_true",
        help="Fetch the full 60-day history instead of the incremental window.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Re-validate tickers before downloading.",
    )
    args = parser.parse_args()

    # Initialise DB
    init_db()

    # Resolve ticker list
    if args.universe == "nifty50":
        tickers = get_nifty50(force_validate=args.validate)
    else:
        tickers = get_nifty100(force_validate=args.validate)

    logger.info("Universe: %d tickers", len(tickers))

    # Always include the index
    all_tickers = tickers + ([INDEX_TICKER] if INDEX_TICKER not in tickers else [])

    days_back = HISTORY_DAYS if args.force_full else FETCH_DAYS
    logger.info(
        "Fetching %d days of data for %d tickers…", days_back, len(all_tickers)
    )

    stored = fetch_and_store(all_tickers, days_back=days_back)
    logger.info("Done. Stored rows for %d tickers.", len(stored))

    print_summary(all_tickers)


if __name__ == "__main__":
    main()
