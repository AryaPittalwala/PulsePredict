"""
replay.py – Replay cursor logic for developing and demoing outside market hours.

The replay cursor steps through stored 5-minute bars one at a time.
At each cursor position, ALL data shown to the UI (movers, charts, predictions)
must use only bars with timestamps ≤ cursor.  Absolutely no lookahead.

A prediction made at cursor position t is resolved when the cursor
reaches t + HORIZON_BARS bars.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import pytz

from config import HORIZON_BARS, TZ
from src.db import query_prices

logger = logging.getLogger(__name__)

IST = pytz.timezone(TZ)


class ReplayCursor:
    """
    Maintains state for a replay session over stored price bars.

    Parameters
    ----------
    tickers  : List of tickers to replay over
    n_days   : Number of recent stored days to replay
    start_ts : Optional explicit start timestamp (IST-aware)
    """

    def __init__(
        self,
        tickers: list[str],
        n_days: int = 5,
        start_ts: Optional[datetime] = None,
    ) -> None:
        self.tickers = tickers
        self.n_days = n_days
        self._timeline: list[datetime] = []
        self._cursor_idx: int = 0
        self._build_timeline(start_ts)

    def _build_timeline(self, start_ts: Optional[datetime]) -> None:
        """
        Build a sorted list of unique 5-min bar timestamps from the DB
        (intersection across the ticker universe).
        """
        if not self.tickers:
            logger.warning("No tickers supplied to ReplayCursor.")
            return

        # Load timestamps for first ticker and use as reference
        df = query_prices(self.tickers[0])
        if df.empty:
            logger.warning("No price data for %s in DB.", self.tickers[0])
            return

        df["ts"] = pd.to_datetime(df["ts"], utc=False)
        if df["ts"].dt.tz is None:
            df["ts"] = df["ts"].dt.tz_localize(TZ)
        else:
            df["ts"] = df["ts"].dt.tz_convert(TZ)

        # Keep only the last n_days of unique trading dates
        dates = df["ts"].dt.date.unique()
        dates_sorted = sorted(dates)
        cutoff_date = dates_sorted[-self.n_days] if len(dates_sorted) >= self.n_days else dates_sorted[0]

        df = df[df["ts"].dt.date >= cutoff_date]
        timestamps = sorted(df["ts"].tolist())

        if start_ts is not None:
            if start_ts.tzinfo is None:
                start_ts = IST.localize(start_ts)
            timestamps = [t for t in timestamps if t >= start_ts]

        self._timeline = timestamps
        logger.info(
            "Replay timeline: %d bars from %s to %s",
            len(self._timeline),
            self._timeline[0] if self._timeline else "?",
            self._timeline[-1] if self._timeline else "?",
        )

    @property
    def current(self) -> Optional[datetime]:
        """Current cursor timestamp, or None if timeline is empty."""
        if not self._timeline:
            return None
        return self._timeline[min(self._cursor_idx, len(self._timeline) - 1)]

    @property
    def at_end(self) -> bool:
        return self._cursor_idx >= len(self._timeline) - 1

    @property
    def at_start(self) -> bool:
        return self._cursor_idx <= 0

    def advance(self, steps: int = 1) -> Optional[datetime]:
        """Move cursor forward by ``steps`` bars.  Returns new current timestamp."""
        self._cursor_idx = min(
            self._cursor_idx + steps, len(self._timeline) - 1
        )
        return self.current

    def rewind(self, steps: int = 1) -> Optional[datetime]:
        """Move cursor backward by ``steps`` bars."""
        self._cursor_idx = max(self._cursor_idx - steps, 0)
        return self.current

    def jump_to(self, ts: datetime) -> Optional[datetime]:
        """Move cursor to the bar closest to ``ts`` without exceeding it."""
        if ts.tzinfo is None:
            ts = IST.localize(ts)
        # Find the rightmost timestamp ≤ ts
        idx = 0
        for i, t in enumerate(self._timeline):
            if t <= ts:
                idx = i
            else:
                break
        self._cursor_idx = idx
        return self.current

    def get_prices_up_to_cursor(
        self, ticker: str, as_of: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Return stored price bars for ``ticker`` up to and including the cursor.

        Guarantees no lookahead: all returned bars have ts ≤ cursor.
        """
        cutoff = as_of or self.current
        if cutoff is None:
            return pd.DataFrame()

        if cutoff.tzinfo is None:
            cutoff = IST.localize(cutoff)

        df = query_prices(ticker, end=cutoff.strftime("%Y-%m-%d %H:%M:%S%z"))
        if df.empty:
            return df

        df["ts"] = pd.to_datetime(df["ts"], utc=False)
        if df["ts"].dt.tz is None:
            df["ts"] = df["ts"].dt.tz_localize(TZ)
        else:
            df["ts"] = df["ts"].dt.tz_convert(TZ)

        # Double-check: drop any rows that sneak past cutoff
        df = df[df["ts"] <= cutoff]
        return df.set_index("ts").sort_index()[["open", "high", "low", "close", "volume"]]

    def horizon_resolve_ts(self) -> Optional[datetime]:
        """
        Timestamp at which predictions made at current cursor are resolved
        (cursor + HORIZON_BARS bars).
        """
        target_idx = self._cursor_idx + HORIZON_BARS
        if target_idx < len(self._timeline):
            return self._timeline[target_idx]
        return None

    @property
    def progress(self) -> float:
        """Fraction of timeline elapsed [0, 1]."""
        if not self._timeline:
            return 0.0
        return self._cursor_idx / (len(self._timeline) - 1)

    def __len__(self) -> int:
        return len(self._timeline)
