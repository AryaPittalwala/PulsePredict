"""
news.py – Google News RSS fetcher for NSE stocks.

Usage
-----
    from src.news import fetch_headlines

Headlines are stored in the ``headlines`` table (db.py).  Call
``sentiment.score_headlines()`` afterwards to populate sentiment fields.
"""

from __future__ import annotations

import logging
import time
import urllib.parse
from datetime import datetime, timezone

import feedparser

from src.db import init_db, upsert_headlines
from src.universe import ticker_to_company

logger = logging.getLogger(__name__)

_RSS_URL = (
    "https://news.google.com/rss/search?"
    "q={query}&hl=en-IN&gl=IN&ceid=IN:en"
)


def _build_url(ticker: str) -> str:
    company = ticker_to_company(ticker)
    query = urllib.parse.quote(f"{company} stock when:1d")
    return _RSS_URL.format(query=query)


def _parse_feed(ticker: str, feed_url: str) -> list[dict]:
    """
    Fetch and parse a Google News RSS feed.

    Returns a list of headline dicts ready for upsert_headlines.
    Returns [] on any error (network, parse, etc.).
    """
    try:
        parsed = feedparser.parse(feed_url)
    except Exception as e:
        logger.warning("feedparser error for %s: %s", ticker, e)
        return []

    if parsed.bozo and not parsed.entries:
        logger.warning("RSS fetch returned bozo for %s: %s", ticker, parsed.bozo_exception)
        return []

    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []
    seen_links: set[str] = set()

    for entry in parsed.entries:
        link = getattr(entry, "link", "") or ""
        title = getattr(entry, "title", "") or ""
        source = ""
        try:
            source = entry.source.title if hasattr(entry, "source") else ""
        except Exception:
            pass

        # Parse published timestamp
        published_ts = ""
        try:
            published_struct = getattr(entry, "published_parsed", None)
            if published_struct:
                published_ts = datetime(*published_struct[:6], tzinfo=timezone.utc).isoformat()
        except Exception:
            pass

        if not link or link in seen_links:
            continue
        seen_links.add(link)

        rows.append(
            {
                "ticker": ticker,
                "published_ts": published_ts,
                "title": title,
                "source": source,
                "link": link,
                "fetched_at": fetched_at,
                "sentiment": None,
                "p_pos": None,
                "p_neg": None,
                "p_neu": None,
            }
        )

    return rows


def fetch_headlines(
    ticker: str,
    sleep_sec: float = 1.0,
    db_init: bool = False,
) -> list[dict]:
    """
    Fetch up-to-24h headlines for ``ticker`` from Google News RSS,
    deduplicate, and store in SQLite.

    Parameters
    ----------
    ticker    : NSE ticker (e.g. "ITC.NS")
    sleep_sec : polite delay before returning (rate-limit safety)
    db_init   : call init_db() first if True

    Returns
    -------
    List of stored headline dicts (may be empty).
    """
    if db_init:
        init_db()

    url = _build_url(ticker)
    logger.info("Fetching headlines for %s from %s", ticker, url)

    rows = _parse_feed(ticker, url)
    if rows:
        n = upsert_headlines(rows)
        logger.info("Stored %d new headlines for %s.", n, ticker)
    else:
        logger.info("No headlines found for %s.", ticker)

    time.sleep(sleep_sec)
    return rows


def fetch_headlines_bulk(
    tickers: list[str],
    sleep_sec: float = 1.5,
) -> dict[str, list[dict]]:
    """
    Fetch headlines for multiple tickers sequentially.

    Returns a dict {ticker: [headlines]}.
    """
    results: dict[str, list[dict]] = {}
    for ticker in tickers:
        results[ticker] = fetch_headlines(ticker, sleep_sec=sleep_sec)
    return results
