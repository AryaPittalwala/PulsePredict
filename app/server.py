"""
server.py – FastAPI backend for NSE Stock Direction Predictor.

Endpoints:
  GET /                        → serves index.html
  GET /api/universe            → NIFTY 50 / 100 ticker list
  GET /api/movers              → top gainers, losers, volume shockers with sparklines
  GET /api/predict             → LightGBM + news-adjusted prediction
  GET /api/chart               → 5-min OHLCV candlestick data
  GET /api/news                → scored headlines for a ticker
  GET /api/scoreboard          → resolved predictions & accuracy metrics
  GET /api/market_status       → NSE open/closed + IST time
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytz
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    INDEX_TICKER,
    INTERVAL,
    MARKET_CLOSE,
    MARKET_OPEN,
    TZ,
)
from src.db import query_prices

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NSE Stock Direction Predictor", version="1.0.0")

IST = pytz.timezone(TZ)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory cache for movers to keep landing page instant
_MOVERS_CACHE: dict = {"timestamp": 0, "data": None}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    open_t = datetime.strptime(MARKET_OPEN, "%H:%M").time()
    close_t = datetime.strptime(MARKET_CLOSE, "%H:%M").time()
    return open_t <= now.time() <= close_t


def _fetch_live_prices(tickers: list[str], period: str = "2d", interval: str = "5m") -> dict[str, pd.DataFrame]:
    """Batch-fetch prices from yfinance with fallback to SQLite."""
    import yfinance as yf
    result = {}
    try:
        batch_str = " ".join(tickers[:20])  # limit batch for speed
        raw = yf.download(batch_str, period=period, interval=interval,
                          group_by="ticker", auto_adjust=True,
                          progress=False, threads=True)
        if not raw.empty:
            for t in tickers[:20]:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        if t in raw.columns.get_level_values(0):
                            df = raw[t].copy()
                        elif t in raw.columns.get_level_values(1):
                            df = raw.xs(t, level=1, axis=1).copy()
                        else:
                            continue
                    else:
                        df = raw.copy()

                    if df.empty:
                        continue
                    df.columns = [str(c).lower() for c in df.columns]
                    df = df.dropna(subset=["close"])
                    if df.index.tzinfo is None:
                        df.index = df.index.tz_localize("UTC")
                    df.index = df.index.tz_convert(TZ)
                    result[t] = df[["open", "high", "low", "close", "volume"]]
                except Exception:
                    continue
    except Exception as e:
        logger.warning("yfinance batch download notice: %s", e)

    # Fill any missing tickers from SQLite cache
    for t in tickers:
        if t not in result or result[t].empty:
            try:
                db_df = query_prices(t)
                if not db_df.empty:
                    db_df["ts"] = pd.to_datetime(db_df["ts"], utc=False)
                    db_df = db_df.set_index("ts").sort_index()
                    result[t] = db_df[["open", "high", "low", "close", "volume"]]
            except Exception:
                pass

    return result


def _get_movers_data(universe: str = "nifty50") -> dict:
    global _MOVERS_CACHE
    now_ts = time.time()
    # Cache for 60 seconds
    if _MOVERS_CACHE["data"] is not None and (now_ts - _MOVERS_CACHE["timestamp"]) < 60:
        return _MOVERS_CACHE["data"]

    from src.universe import get_nifty50, get_nifty100, ticker_to_company
    nifty50 = get_nifty50()
    all_tickers = get_nifty100()
    target_tickers = nifty50[:15] if universe == "nifty50" else all_tickers[:15]

    prices = _fetch_live_prices(target_tickers, period="2d", interval="5m")

    rows = []
    for ticker, df in prices.items():
        if df.empty or len(df) < 2:
            continue
        try:
            df = df.sort_index()
            # Select the most recent trading session in the data
            dates = df.index.date
            latest_date = dates.max()
            session_df = df[dates == latest_date]
            if session_df.empty:
                session_df = df.tail(30)

            # Clean session df
            session_df = session_df.dropna(subset=["open", "close"])
            if session_df.empty or len(session_df) < 2:
                continue

            day_open = float(session_df.iloc[0]["open"])
            last_close = float(session_df.iloc[-1]["close"])
            if np.isnan(day_open) or np.isnan(last_close) or day_open == 0:
                continue

            change = last_close - day_open
            change_pct = (change / day_open) * 100
            if np.isnan(change_pct):
                change_pct = 0.0

            volume_val = session_df["volume"].fillna(0).sum()
            volume = int(volume_val) if not np.isnan(volume_val) else 0

            raw_sparkline = session_df["close"].dropna().tolist()[-30:]
            sparkline = [round(float(x), 2) for x in raw_sparkline if not np.isnan(float(x))]
            if len(sparkline) < 2:
                sparkline = [round(last_close, 2), round(last_close, 2)]

            rows.append({
                "ticker": ticker,
                "name": ticker_to_company(ticker),
                "price": round(last_close, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "volume": volume,
                "sparkline": sparkline,
                "is_positive": change_pct >= 0,
            })
        except Exception as e:
            logger.debug("Row compute error for %s: %s", ticker, e)
            continue

    rows.sort(key=lambda x: x["change_pct"], reverse=True)
    gainers = [r for r in rows if r["is_positive"]][:5]
    if not gainers:
        gainers = rows[:5]

    losers = sorted([r for r in rows if not r["is_positive"]], key=lambda x: x["change_pct"])[:5]
    if not losers:
        losers = sorted(rows, key=lambda x: x["change_pct"])[:5]

    vol_sorted = sorted(rows, key=lambda x: x["volume"], reverse=True)[:5]

    res = {"gainers": gainers, "losers": losers, "volume_shockers": vol_sorted}
    _MOVERS_CACHE = {"timestamp": now_ts, "data": res}
    return res


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return JSONResponse({"error": "Frontend not found. static/index.html is missing."}, status_code=404)


@app.get("/api/market_status")
async def market_status():
    now = datetime.now(IST)
    is_open = _is_market_open()
    return {
        "is_open": is_open,
        "status": "Live" if is_open else "Closed",
        "time": now.strftime("%H:%M:%S IST"),
        "date": now.strftime("%d %b %Y"),
        "day": now.strftime("%A"),
    }


@app.get("/api/universe")
async def get_universe():
    from src.universe import get_nifty50, get_nifty100, ticker_to_company
    all_tickers = get_nifty100()
    nifty50 = get_nifty50()
    return {
        "nifty50": [{"ticker": t, "name": ticker_to_company(t)} for t in nifty50],
        "nifty100": [{"ticker": t, "name": ticker_to_company(t)} for t in all_tickers],
    }


@app.get("/api/movers")
async def get_movers(universe: str = Query(default="nifty50", enum=["nifty50", "nifty100"])):
    try:
        data = _get_movers_data(universe)
        return data
    except Exception as e:
        logger.error("Movers error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chart")
async def get_chart(
    ticker: str = Query(..., description="e.g. RELIANCE.NS"),
    mode: str = Query(default="live", enum=["live", "replay"]),
    as_of: Optional[str] = Query(default=None),
    days: int = Query(default=5),
):
    try:
        df = pd.DataFrame()
        if mode == "live":
            prices = _fetch_live_prices([ticker], period=f"{max(days, 5)}d", interval="5m")
            df = prices.get(ticker, pd.DataFrame())

        if df.empty:
            # Fallback to SQLite cache
            as_of_dt = pd.to_datetime(as_of) if as_of else datetime.now(IST)
            if as_of_dt.tzinfo is None:
                as_of_dt = IST.localize(as_of_dt)
            start = (as_of_dt - timedelta(days=days + 2)).strftime("%Y-%m-%d %H:%M:%S%z")
            end = as_of_dt.strftime("%Y-%m-%d %H:%M:%S%z")
            df_raw = query_prices(ticker, start=start, end=end)
            if not df_raw.empty:
                df_raw["ts"] = pd.to_datetime(df_raw["ts"])
                df = df_raw.set_index("ts").sort_index()

        if df.empty:
            # Last-ditch: get any recent bars for ticker from SQLite
            df_raw = query_prices(ticker)
            if not df_raw.empty:
                df_raw["ts"] = pd.to_datetime(df_raw["ts"])
                df = df_raw.set_index("ts").sort_index().tail(100)

        if df.empty:
            return {"ticker": ticker, "candles": [], "volumes": []}

        df = df.dropna(subset=["open", "high", "low", "close"]).sort_index()
        candles = []
        volumes = []
        for ts, row in df.iterrows():
            try:
                ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S") if hasattr(ts, "strftime") else str(ts)
                o = float(row.get("open", 0))
                h = float(row.get("high", 0))
                l = float(row.get("low", 0))
                c = float(row.get("close", 0))
                v = float(row.get("volume", 0))
                if np.isnan(o) or np.isnan(h) or np.isnan(l) or np.isnan(c):
                    continue
                candles.append({
                    "t": ts_str,
                    "o": round(o, 2),
                    "h": round(h, 2),
                    "l": round(l, 2),
                    "c": round(c, 2),
                })
                volumes.append({"t": ts_str, "v": int(v) if not np.isnan(v) else 0})
            except Exception:
                continue

        return {"ticker": ticker, "candles": candles, "volumes": volumes}
    except Exception as e:
        logger.error("Chart error for %s: %s", ticker, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict")
async def get_prediction(
    ticker: str = Query(...),
    mode: str = Query(default="live", enum=["live", "replay"]),
    as_of: Optional[str] = Query(default=None),
    include_news: bool = Query(default=True),
    news_weight: float = Query(default=0.35),
):
    try:
        from src.predict import predict_for
        from src.scoreboard import log_prediction

        as_of_dt = None
        if as_of:
            as_of_dt = pd.to_datetime(as_of)
            if as_of_dt.tzinfo is None:
                as_of_dt = IST.localize(as_of_dt)

        result = predict_for(
            ticker=ticker,
            as_of=as_of_dt,
            mode=mode,
            include_news=include_news,
            news_weight=news_weight,
        )

        # Log prediction for automated resolution
        try:
            result_for_log = dict(result)
            result_for_log["mode"] = mode
            log_prediction(result_for_log)
        except Exception as log_err:
            logger.debug("Prediction logging note: %s", log_err)

        # Flatten news sentiment helper flags for frontend
        news_info = result.get("news_sentiment", {})
        result["sentiment_applied"] = news_info.get("sentiment_applied", False)
        result["shift_summary"] = news_info.get("shift_summary")

        return result
    except Exception as e:
        logger.error("Prediction error for %s: %s", ticker, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news")
async def get_news(ticker: str = Query(...)):
    try:
        from src.news import fetch_headlines
        from src.sentiment import aggregate_sentiment, score_headlines
        try:
            fetch_headlines(ticker, sleep_sec=0.1)
            score_headlines(batch_size=16)
        except Exception as e:
            logger.debug("Live news fetch notice: %s", e)

        agg = aggregate_sentiment(ticker, window_hours=24)
        return agg
    except Exception as e:
        logger.error("News error for %s: %s", ticker, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scoreboard")
async def get_scoreboard(mode: Optional[str] = Query(default=None)):
    try:
        from src.scoreboard import get_scoreboard as fetch_sb, resolve_pending_predictions
        try:
            resolve_pending_predictions(mode=mode)
        except Exception as res_err:
            logger.debug("Scoreboard resolution notice: %s", res_err)

        return fetch_sb(mode=mode)
    except Exception as e:
        logger.error("Scoreboard error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
