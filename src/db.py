"""
db.py – SQLite helpers: table creation, upsert, and query utilities.

All schema definitions live here.  Import this module and call
``init_db()`` once at startup.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from typing import Generator

import pandas as pd

from config import DB_PATH

logger = logging.getLogger(__name__)

# ── Schema DDL ────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS prices (
    ticker  TEXT    NOT NULL,
    ts      TEXT    NOT NULL,
    open    REAL,
    high    REAL,
    low     REAL,
    close   REAL,
    volume  REAL,
    PRIMARY KEY (ticker, ts)
);

CREATE INDEX IF NOT EXISTS idx_prices_ticker_ts ON prices (ticker, ts);

CREATE TABLE IF NOT EXISTS headlines (
    ticker        TEXT    NOT NULL,
    published_ts  TEXT,
    title         TEXT,
    source        TEXT,
    link          TEXT    NOT NULL,
    fetched_at    TEXT,
    sentiment     REAL,
    p_pos         REAL,
    p_neg         REAL,
    p_neu         REAL,
    PRIMARY KEY (ticker, link)
);

CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT,
    made_at       TEXT,
    horizon_bars  INTEGER,
    pred_label    TEXT,
    p_up          REAL,
    p_down        REAL,
    p_neutral     REAL,
    confidence    REAL,
    mode          TEXT,
    resolved      INTEGER DEFAULT 0,
    actual_ret    REAL,
    actual_label  TEXT,
    correct       INTEGER
);

CREATE INDEX IF NOT EXISTS idx_pred_ticker_made ON predictions (ticker, made_at);
"""


def init_db(db_path: str = DB_PATH) -> None:
    """Create all tables if they do not already exist."""
    import os
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with connect(db_path) as conn:
        conn.executescript(_DDL)
    logger.info("Database initialised at %s", db_path)


@contextmanager
def connect(db_path: str = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that opens a WAL-mode SQLite connection."""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Prices ────────────────────────────────────────────────────────────────────

def upsert_prices(df: pd.DataFrame, db_path: str = DB_PATH) -> int:
    """
    Insert or replace rows into the ``prices`` table.

    Parameters
    ----------
    df : DataFrame with columns [ticker, ts, open, high, low, close, volume].
         ``ts`` must be a timezone-aware string (IST).

    Returns
    -------
    Number of rows inserted/replaced.
    """
    if df.empty:
        return 0
    rows = df[["ticker", "ts", "open", "high", "low", "close", "volume"]].to_records(
        index=False
    )
    with connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO prices (ticker, ts, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [tuple(r) for r in rows],
        )
    return len(rows)


def query_prices(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """
    Load price rows for a ticker from SQLite.

    Parameters
    ----------
    ticker : NSE ticker string
    start  : ISO-format lower bound (inclusive), optional
    end    : ISO-format upper bound (inclusive), optional

    Returns
    -------
    DataFrame indexed by ``ts`` (string), sorted ascending.
    """
    sql = "SELECT ticker, ts, open, high, low, close, volume FROM prices WHERE ticker = ?"
    params: list = [ticker]
    if start:
        sql += " AND ts >= ?"
        params.append(start)
    if end:
        sql += " AND ts <= ?"
        params.append(end)
    sql += " ORDER BY ts ASC"

    with connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    return df


def query_prices_multi(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """Load prices for multiple tickers at once."""
    placeholders = ",".join("?" * len(tickers))
    sql = f"SELECT ticker, ts, open, high, low, close, volume FROM prices WHERE ticker IN ({placeholders})"
    params: list = list(tickers)
    if start:
        sql += " AND ts >= ?"
        params.append(start)
    if end:
        sql += " AND ts <= ?"
        params.append(end)
    sql += " ORDER BY ticker, ts ASC"

    with connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    return df


def get_ticker_date_range(
    ticker: str, db_path: str = DB_PATH
) -> tuple[str | None, str | None]:
    """Return (min_ts, max_ts) for a ticker, or (None, None) if no data."""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT MIN(ts), MAX(ts) FROM prices WHERE ticker = ?", (ticker,)
        ).fetchone()
    return row if row and row[0] else (None, None)


def get_ticker_row_count(ticker: str, db_path: str = DB_PATH) -> int:
    """Return the number of price rows stored for a ticker."""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM prices WHERE ticker = ?", (ticker,)
        ).fetchone()
    return row[0] if row else 0


def list_tickers(db_path: str = DB_PATH) -> list[str]:
    """Return all distinct tickers present in the prices table."""
    with connect(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT ticker FROM prices ORDER BY ticker").fetchall()
    return [r[0] for r in rows]


# ── Headlines ────────────────────────────────────────────────────────────────

def upsert_headlines(rows: list[dict], db_path: str = DB_PATH) -> int:
    """Insert or ignore headline rows (PRIMARY KEY deduplicates)."""
    if not rows:
        return 0
    with connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO headlines
              (ticker, published_ts, title, source, link, fetched_at,
               sentiment, p_pos, p_neg, p_neu)
            VALUES (:ticker, :published_ts, :title, :source, :link, :fetched_at,
                    :sentiment, :p_pos, :p_neg, :p_neu)
            """,
            rows,
        )
    return len(rows)


def update_headline_sentiment(
    link: str,
    sentiment: float,
    p_pos: float,
    p_neg: float,
    p_neu: float,
    db_path: str = DB_PATH,
) -> None:
    """Update sentiment scores for a single headline row."""
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE headlines SET sentiment=?, p_pos=?, p_neg=?, p_neu=?
            WHERE link=?
            """,
            (sentiment, p_pos, p_neg, p_neu, link),
        )


def query_headlines(
    ticker: str,
    window_hours: int = 24,
    as_of: str | datetime | None = None,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """Return headlines for ``ticker`` within ``window_hours`` up to ``as_of``."""
    sql = """
        SELECT ticker, published_ts, title, source, link, fetched_at,
               sentiment, p_pos, p_neg, p_neu
        FROM headlines
        WHERE ticker = ?
    """
    params = [ticker]
    if as_of:
        as_of_str = as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of)
        sql += " AND fetched_at <= ? AND fetched_at >= datetime(?, ?)"
        params.extend([as_of_str, as_of_str, f"-{window_hours} hours"])
    else:
        sql += " AND fetched_at >= datetime('now', ?)"
        params.append(f"-{window_hours} hours")
    sql += " ORDER BY published_ts DESC"
    with connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    return df


def query_unscored_headlines(db_path: str = DB_PATH) -> pd.DataFrame:
    """Return all headlines that have not yet been scored by FinBERT."""
    sql = """
        SELECT ticker, link, title FROM headlines
        WHERE sentiment IS NULL
        ORDER BY fetched_at DESC
    """
    with connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn)
    return df


# ── Predictions ──────────────────────────────────────────────────────────────

def insert_prediction(row: dict, db_path: str = DB_PATH) -> int:
    """Insert a prediction row and return the new id."""
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
              (ticker, made_at, horizon_bars, pred_label, p_up, p_down, p_neutral,
               confidence, mode, resolved)
            VALUES (:ticker, :made_at, :horizon_bars, :pred_label, :p_up, :p_down,
                    :p_neutral, :confidence, :mode, 0)
            """,
            row,
        )
        return cur.lastrowid


def get_unresolved_predictions(
    mode: str | None = None, db_path: str = DB_PATH
) -> pd.DataFrame:
    """Return predictions that have not yet been resolved."""
    sql = "SELECT * FROM predictions WHERE resolved = 0"
    params: list = []
    if mode:
        sql += " AND mode = ?"
        params.append(mode)
    with connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    return df


def resolve_prediction(
    pred_id: int,
    actual_ret: float,
    actual_label: str,
    correct: int,
    db_path: str = DB_PATH,
) -> None:
    """Mark a prediction as resolved with actual outcome."""
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE predictions
            SET resolved=1, actual_ret=?, actual_label=?, correct=?
            WHERE id=?
            """,
            (actual_ret, actual_label, correct, pred_id),
        )


def query_predictions(
    mode: str | None = None,
    resolved_only: bool = False,
    limit: int = 100,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """Query prediction history."""
    sql = "SELECT * FROM predictions WHERE 1=1"
    params: list = []
    if mode:
        sql += " AND mode = ?"
        params.append(mode)
    if resolved_only:
        sql += " AND resolved = 1"
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    return df
