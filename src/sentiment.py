"""
sentiment.py – FinBERT sentiment scoring and aggregation.

FinBERT is loaded once per process (cached).  Headlines that have already
been scored are not re-scored.

Sentiment score = p_pos - p_neg ∈ [-1, 1].

Usage
-----
    from src.sentiment import score_headlines, aggregate_sentiment
    score_headlines()  # score all unscored rows in DB
    agg = aggregate_sentiment("ITC.NS")
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional

import pandas as pd

from config import FINBERT_MODEL
from src.db import query_headlines, query_unscored_headlines, update_headline_sentiment

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict = {}


def _load_finbert():
    """Load FinBERT tokenizer and model (cached after first call)."""
    if "pipeline" in _MODEL_CACHE:
        return _MODEL_CACHE["pipeline"]

    logger.info("Loading FinBERT model '%s' (this may take a moment)…", FINBERT_MODEL)
    try:
        from transformers import pipeline as hf_pipeline

        pipe = hf_pipeline(
            "text-classification",
            model=FINBERT_MODEL,
            tokenizer=FINBERT_MODEL,
            top_k=None,          # return all labels
            device=-1,           # CPU
            truncation=True,
            max_length=512,
        )
        _MODEL_CACHE["pipeline"] = pipe
        logger.info("FinBERT loaded successfully.")
        return pipe
    except Exception as e:
        logger.error("Failed to load FinBERT: %s", e)
        return None


def _lexicon_fallback(text: str) -> dict:
    """Heuristic fallback when FinBERT is unavailable or downloading."""
    t = text.lower()
    pos = ["jump", "rise", "gain", "surged", "surge", "high", "rally", "profit", "bull", "growth", "boost", "outperform", "beat"]
    neg = ["drop", "fall", "loss", "plunge", "decline", "slump", "crash", "bear", "miss", "warn", "debt", "risk", "cut"]
    p_count = sum(1 for w in pos if f" {w} " in f" {t} " or t.startswith(f"{w} "))
    n_count = sum(1 for w in neg if f" {w} " in f" {t} " or t.startswith(f"{w} "))
    if p_count > n_count:
        return {"sentiment": 0.5, "p_pos": 0.7, "p_neg": 0.2, "p_neu": 0.1}
    elif n_count > p_count:
        return {"sentiment": -0.5, "p_pos": 0.2, "p_neg": 0.7, "p_neu": 0.1}
    else:
        return {"sentiment": 0.0, "p_pos": 0.15, "p_neg": 0.15, "p_neu": 0.7}


def _score_texts(texts: list[str], batch_size: int = 32) -> list[dict]:
    """
    Score a list of text strings with FinBERT.

    Returns a list of dicts with keys: sentiment, p_pos, p_neg, p_neu.
    Falls back to lexicon-based scores on failure.
    """
    pipe = _load_finbert()
    if pipe is None:
        logger.info("FinBERT unavailable — using financial lexicon fallback.")
        return [_lexicon_fallback(t) for t in texts]

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            outputs = pipe(batch)
            for out in outputs:
                label_scores = {item["label"].lower(): item["score"] for item in out}
                p_pos = label_scores.get("positive", 0.0)
                p_neg = label_scores.get("negative", 0.0)
                p_neu = label_scores.get("neutral", 1.0)
                # Normalise (they should sum to ~1 already)
                total = p_pos + p_neg + p_neu
                if total > 0:
                    p_pos, p_neg, p_neu = p_pos / total, p_neg / total, p_neu / total
                results.append(
                    {
                        "sentiment": float(p_pos - p_neg),
                        "p_pos": float(p_pos),
                        "p_neg": float(p_neg),
                        "p_neu": float(p_neu),
                    }
                )
        except Exception as e:
            logger.warning("FinBERT batch error: %s", e)
            results.extend(
                [{"sentiment": 0.0, "p_pos": 0.0, "p_neg": 0.0, "p_neu": 1.0}] * len(batch)
            )
    return results


def score_headlines(batch_size: int = 32) -> int:
    """
    Score all unscored headlines in the DB with FinBERT.

    Returns the number of headlines scored.
    """
    df = query_unscored_headlines()
    if df.empty:
        logger.info("No unscored headlines.")
        return 0

    logger.info("Scoring %d headlines with FinBERT…", len(df))
    texts = df["title"].fillna("").tolist()
    scores = _score_texts(texts, batch_size=batch_size)

    for (_, row), score in zip(df.iterrows(), scores):
        update_headline_sentiment(
            link=row["link"],
            sentiment=score["sentiment"],
            p_pos=score["p_pos"],
            p_neg=score["p_neg"],
            p_neu=score["p_neu"],
        )

    logger.info("Scored %d headlines.", len(df))
    return len(df)


def aggregate_sentiment(
    ticker: str,
    window_hours: int = 24,
    as_of: Optional[datetime] = None,
) -> dict:
    """
    Return aggregated sentiment for ``ticker`` over the last ``window_hours`` up to ``as_of``.

    Returns
    -------
    dict with keys:
      - mean_sentiment  : float in [-1, 1] (or None if no data)
      - headline_count  : int
      - latest_headlines: list of {title, source, published_ts, sentiment}
    """
    df = query_headlines(ticker, window_hours=window_hours, as_of=as_of)

    if df.empty:
        return {
            "mean_sentiment": None,
            "headline_count": 0,
            "latest_headlines": [],
        }

    scored = df.dropna(subset=["sentiment"])
    mean_sent = float(scored["sentiment"].mean()) if not scored.empty else None

    latest = df.head(5)[["title", "source", "published_ts", "sentiment"]].to_dict("records")

    return {
        "mean_sentiment": mean_sent,
        "headline_count": len(df),
        "latest_headlines": latest,
    }


def get_sentiment_features_for_ticker(
    ticker: str,
    as_of: Optional[datetime] = None,
    window_hours: int = 24,
) -> pd.Series:
    """
    Return per-bar sentiment feature series aligned to a time index.

    This is used when USE_SENTIMENT_FEATURES is True.  As_of is used in
    replay mode to avoid lookahead.
    """
    from src.db import connect
    from config import DB_PATH

    sql = """
        SELECT fetched_at, sentiment FROM headlines
        WHERE ticker = ? AND sentiment IS NOT NULL
    """
    params: list = [ticker]
    if as_of:
        sql += " AND fetched_at <= ?"
        params.append(as_of.isoformat())
    sql += " ORDER BY fetched_at ASC"

    with connect(DB_PATH) as conn:
        df = pd.read_sql_query(sql, conn, params=params)

    if df.empty:
        return pd.Series(dtype=float, name="sent_24h_mean")

    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    df = df.set_index("fetched_at").sort_index()
    return df["sentiment"]
