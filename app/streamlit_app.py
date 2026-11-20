"""
streamlit_app.py – NSE Intraday Stock Direction Predictor Dashboard.

Layout
------
1. Header + sidebar: title, disclaimer, mode toggle, universe selector,
   refresh, replay controls, market status banner.
2. Top movers table: Gainers / Losers / Volume shockers with sparklines.
3. Stock detail panel: candlestick chart, prediction card, news, scoreboard.

Performance
-----------
- Prices for the whole universe are fetched in batches and cached with
  st.cache_data(ttl=120).
- Falls back to SQLite cache on yfinance failure.
- Never blocks the UI on 100 sequential downloads.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    CLASS_DOWN,
    CLASS_NEUTRAL,
    CLASS_UP,
    CONFIDENCE_THRESHOLD,
    HORIZON_BARS,
    INDEX_TICKER,
    INTERVAL,
    MARKET_CLOSE,
    MARKET_OPEN,
    TZ,
)

import urllib.parse

logger = logging.getLogger(__name__)
IST = pytz.timezone(TZ)

# ── Custom Modern CSS ────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    /* Sleek card container */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .badge-up {
        background: rgba(0, 230, 118, 0.15);
        color: #00E676;
        border: 1px solid #00E676;
        border-radius: 8px;
        padding: 4px 14px;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    .badge-down {
        background: rgba(255, 82, 82, 0.15);
        color: #FF5252;
        border: 1px solid #FF5252;
        border-radius: 8px;
        padding: 4px 14px;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    .badge-neutral {
        background: rgba(176, 190, 197, 0.15);
        color: #B0BEC5;
        border: 1px solid #78909C;
        border-radius: 8px;
        padding: 4px 14px;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    /* Dataframe rounded styling */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
"""

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Stock Direction Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Disclaimer ────────────────────────────────────────────────────────────────
DISCLAIMER = (
    "⚠️ **Educational purpose only — NOT financial advice.** "
    "This tool is a machine-learning experiment. Predictions are often wrong. "
    "Never make investment decisions based on this output. "
    "Short-horizon stock prediction is extremely difficult; markets are "
    "approximately efficient. See the README for methodology and limitations."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    open_t = datetime.strptime(MARKET_OPEN, "%H:%M").time()
    close_t = datetime.strptime(MARKET_CLOSE, "%H:%M").time()
    return open_t <= now.time() <= close_t


@st.cache_data(ttl=120, show_spinner=False)
def _fetch_universe_prices(tickers: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """
    Download today's 5-minute bars for all tickers in batches.
    Falls back to SQLite on failure.
    """
    import yfinance as yf
    from config import DOWNLOAD_BATCH_SIZE, DOWNLOAD_SLEEP_SEC
    from src.db import query_prices

    result: dict[str, pd.DataFrame] = {}
    ticker_list = list(tickers)

    for i in range(0, len(ticker_list), DOWNLOAD_BATCH_SIZE):
        batch = ticker_list[i : i + DOWNLOAD_BATCH_SIZE]
        try:
            raw = yf.download(
                batch,
                period="1d",
                interval="5m",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                raise ValueError("Empty response")
            for t in batch:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        if t in raw.columns.get_level_values(0):
                            df = raw[t].copy()
                        elif t in raw.columns.get_level_values(1):
                            df = raw.xs(t, level=1, axis=1).copy()
                        elif "Price" in raw.columns.names and len(batch) == 1:
                            df = raw.copy()
                            df.columns = df.columns.get_level_values("Price")
                        else:
                            continue
                    else:
                        df = raw.copy()

                    if df.empty:
                        continue
                    if df.index.tzinfo is None:
                        df.index = df.index.tz_localize("UTC")
                    df.index = df.index.tz_convert(TZ)
                    df.columns = [str(c).lower() for c in df.columns]
                    result[t] = df[["open", "high", "low", "close", "volume"]]
                except Exception:
                    pass
        except Exception as e:
            logger.warning("yfinance batch failed (%s). Falling back to SQLite.", e)
            for t in batch:
                try:
                    df = query_prices(t)
                    if not df.empty:
                        df["ts"] = pd.to_datetime(df["ts"], utc=False)
                        df = df.set_index("ts").sort_index()
                        result[t] = df[["open", "high", "low", "close", "volume"]]
                except Exception:
                    pass
        time.sleep(0.3)

    return result


def _make_svg_sparkline(prices: list[float], is_positive: bool, width: int = 140, height: int = 40) -> str:
    """
    Generate a crisp SVG sparkline encoded as a data URI for st.column_config.ImageColumn.
    Green (#00E676) for positive/gain, Red (#FF5252) for loss/negative.
    """
    clean = []
    for v in prices:
        try:
            val = float(v)
            if not math.isnan(val):
                clean.append(val)
        except (ValueError, TypeError):
            continue

    if len(clean) < 2:
        clean = [100.0, 100.0]

    mn, mx = min(clean), max(clean)
    rng = mx - mn if mx != mn else 1.0
    pad_x, pad_y = 5, 5

    pts = []
    n = len(clean)
    for i, v in enumerate(clean):
        x = pad_x + (i / (n - 1)) * (width - 2 * pad_x)
        y = (height - pad_y) - ((v - mn) / rng) * (height - 2 * pad_y)
        pts.append(f"{x:.1f},{y:.1f}")

    pts_str = " ".join(pts)
    last_x, last_y = pts[-1].split(",")

    # Green for positive, Red for negative
    stroke_color = "#00E676" if is_positive else "#FF5252"
    fill_color = "rgba(0, 230, 118, 0.16)" if is_positive else "rgba(255, 82, 82, 0.16)"

    poly_pts = f"{pad_x:.1f},{height} " + pts_str + f" {width - pad_x:.1f},{height}"
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<polygon points="{poly_pts}" fill="{fill_color}"/>'
        f'<polyline points="{pts_str}" fill="none" stroke="{stroke_color}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="{last_x}" cy="{last_y}" r="3" fill="{stroke_color}"/>'
        f'</svg>'
    )
    encoded = urllib.parse.quote(svg)
    return f"data:image/svg+xml;utf8,{encoded}"


def _get_movers_df(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute gainers/losers/volume shockers from today's price data."""
    rows = []
    for ticker, df in prices.items():
        if df.empty or len(df) < 2:
            continue
        last = df.iloc[-1]
        first = df.iloc[0]
        prev_close = float(first["open"])
        last_price = float(last["close"])
        change = last_price - prev_close
        pct_change = change / prev_close * 100 if prev_close != 0 else 0
        volume = float(df["volume"].sum())
        sparkline = df["close"].tolist()
        is_positive = (change >= 0)
        sparkline_uri = _make_svg_sparkline(sparkline, is_positive=is_positive)

        rows.append(
            {
                "Ticker": ticker.replace(".NS", ""),
                "_ticker": ticker,
                "Last Price": round(last_price, 2),
                "Change": round(change, 2),
                "Change %": round(pct_change, 2),
                "Volume": int(volume),
                "Today": sparkline_uri,
            }
        )
    if not rows:
        return pd.DataFrame()
    df_out = pd.DataFrame(rows)
    return df_out


def _prediction_color(label: str) -> str:
    return {"UP": "#00cc66", "DOWN": "#ff4444", "NEUTRAL": "#aaaaaa"}.get(label, "#aaaaaa")


def _sentiment_badge(score: Optional[float]) -> str:
    if score is None:
        return "⬜ No sentiment data"
    if score > 0.1:
        return f"🟢 Positive ({score:+.2f})"
    if score < -0.1:
        return f"🔴 Negative ({score:+.2f})"
    return f"🟡 Neutral ({score:+.2f})"


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render sidebar controls and return configuration dict."""
    st.sidebar.title("⚙️ Controls")

    st.sidebar.markdown(DISCLAIMER)
    st.sidebar.markdown("---")

    mode = st.sidebar.radio("Mode", ["Live", "Replay"], index=0)
    universe = st.sidebar.radio("Universe", ["NIFTY 50", "NIFTY 100"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**News & Sentiment Impact**")
    include_news = st.sidebar.checkbox(
        "📰 Factor in Live News",
        value=True,
        help="When enabled, breaking live news (e.g. CEO fired, regulatory actions) dynamically shifts the 15-minute prediction.",
    )
    news_weight = st.sidebar.slider(
        "News Influence Weight",
        min_value=0.10,
        max_value=0.60,
        value=0.35,
        step=0.05,
        help="Controls how strongly breaking news can swing the final direction relative to historical technical indicators.",
    )

    refresh = st.sidebar.button("🔄 Refresh")

    replay_config = {}
    if mode == "Replay":
        st.sidebar.markdown("**Replay Controls**")
        speed = st.sidebar.slider("Auto-advance speed (bars/sec)", 0, 10, 0)
        next_bar = st.sidebar.button("⏭ Next bar")
        replay_config = {"speed": speed, "next_bar": next_bar}

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"🕐 IST: `{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}`"
    )

    return {
        "mode": mode,
        "universe": universe,
        "refresh": refresh,
        "replay": replay_config,
        "include_news": include_news,
        "news_weight": news_weight,
    }


# ── Movers table ──────────────────────────────────────────────────────────────

def render_movers_table(prices: dict[str, pd.DataFrame]) -> Optional[str]:
    """
    Render the top movers table with tabs (Gainers/Losers/Volume).
    Returns the selected ticker or None.
    """
    movers = _get_movers_df(prices)
    if movers.empty:
        st.warning("No price data available. Check your connection or wait for market hours.")
        return None

    tab_gainers, tab_losers, tab_volume = st.tabs(["📈 Gainers", "📉 Losers", "🔥 Volume Shockers"])

    selected_ticker: Optional[str] = None

    with tab_gainers:
        top = movers.nlargest(10, "Change %").reset_index(drop=True)
        sel = _render_table(top, "gainers")
        if sel:
            selected_ticker = sel

    with tab_losers:
        bot = movers.nsmallest(10, "Change %").reset_index(drop=True)
        sel = _render_table(bot, "losers")
        if sel:
            selected_ticker = sel

    with tab_volume:
        vol = movers.nlargest(10, "Volume").reset_index(drop=True)
        sel = _render_table(vol, "volume")
        if sel:
            selected_ticker = sel

    return selected_ticker


def _render_table(df: pd.DataFrame, key: str) -> Optional[str]:
    """Render a movers sub-table with row selection. Returns selected _ticker."""
    if df.empty:
        st.info("No data.")
        return None

    display_df = df[["Ticker", "Last Price", "Change", "Change %", "Volume", "Today"]].copy()

    event = st.dataframe(
        display_df,
        column_config={
            "Today": st.column_config.ImageColumn(
                "Today (Trend)",
                width="small",
                help="Green = Intraday Gain, Red = Intraday Loss",
            ),
            "Change %": st.column_config.NumberColumn(format="%.2f%%"),
            "Volume": st.column_config.NumberColumn(format="%d"),
            "Last Price": st.column_config.NumberColumn(format="₹%.2f"),
            "Change": st.column_config.NumberColumn(format="%+.2f"),
        },
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"movers_{key}",
    )
    sel = event.selection.rows if event.selection else []
    if sel:
        return df.iloc[sel[0]]["_ticker"]
    return None


# ── Candlestick chart ─────────────────────────────────────────────────────────

def render_candlestick(ticker: str, df: pd.DataFrame) -> None:
    if df.empty:
        st.warning(f"No price data for {ticker}.")
        return

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#00E676",
            increasing_fillcolor="#00E676",
            decreasing_line_color="#FF5252",
            decreasing_fillcolor="#FF5252",
        )
    )

    # Volume bar chart on secondary y-axis
    colors = [
        "#00E676" if c >= o else "#FF5252"
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.35,
            yaxis="y2",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title=f"<b>{ticker}</b> — Intraday 5-min Session",
        xaxis_title="Time (IST)",
        yaxis_title="Price (₹)",
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        xaxis_rangeslider_visible=False,
        height=470,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", y=1.03, x=0.01),
        plot_bgcolor="rgba(14, 17, 23, 0.5)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Prediction card ───────────────────────────────────────────────────────────

def render_prediction(result: dict) -> None:
    label = result.get("label", "NEUTRAL")
    probs = result.get("probs", {"UP": 0.33, "DOWN": 0.33, "NEUTRAL": 0.34})
    conf = result.get("confidence", 0.0)
    as_of = result.get("as_of", "")
    horizon = result.get("horizon_minutes", 15)
    err = result.get("error")

    color = _prediction_color(label)

    if err:
        st.warning(f"Prediction unavailable: {err}")
        return

    col1, col2, col3 = st.columns([1, 2, 1])

    badge_class = f"badge-{label.lower()}"
    with col1:
        st.markdown(
            f"""
            <div style='text-align:center; padding: 4px;'>
                <div class='{badge_class}' style='font-size: 2.2em; padding: 6px 20px; margin-bottom: 8px;'>{label}</div>
                <div style='color: #90A4AE; font-size: 0.95em;'>Confidence: <b>{conf:.1%}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("**Probabilities**")
        for cls, p in sorted(probs.items(), key=lambda x: -x[1]):
            st.progress(p, text=f"{cls}: {p:.1%}")

    with col3:
        st.metric("Horizon", f"{horizon} min", help="15-minute lookahead")
        st.caption(f"As of: {as_of}")
        if conf >= CONFIDENCE_THRESHOLD:
            st.success("✓ Above confidence threshold")
        else:
            st.info("Below confidence threshold — low conviction prediction")

    news_info = result.get("news_sentiment", {})
    sentiment_applied = news_info.get("sentiment_applied", False)
    raw_label = result.get("raw_label", label)
    raw_probs = result.get("raw_probs", probs)
    shift_summary = news_info.get("shift_summary", "")

    if sentiment_applied:
        st.info(
            f"📰 **Live News Adjustment Applied:** {shift_summary}\n\n"
            f"• **Technical Indicator Baseline:** `{raw_label}` (UP: {raw_probs.get('UP', 0):.1%}, DOWN: {raw_probs.get('DOWN', 0):.1%}, NEUTRAL: {raw_probs.get('NEUTRAL', 0):.1%})\n\n"
            f"• **Composite Prediction:** `{label}` (UP: {probs.get('UP', 0):.1%}, DOWN: {probs.get('DOWN', 0):.1%}, NEUTRAL: {probs.get('NEUTRAL', 0):.1%})"
        )

    st.caption(
        "⚠️ This is a model estimate, not investment advice. "
        "Accuracy is not guaranteed. See disclaimer above."
    )


# ── News panel ────────────────────────────────────────────────────────────────

def render_news(ticker: str) -> None:
    try:
        from src.news import fetch_headlines
        from src.sentiment import aggregate_sentiment, score_headlines

        with st.spinner("Fetching news…"):
            fetch_headlines(ticker)
            score_headlines(batch_size=16)
            agg = aggregate_sentiment(ticker)

        st.markdown(f"**Sentiment:** {_sentiment_badge(agg['mean_sentiment'])}")
        st.caption(f"{agg['headline_count']} headline(s) in last 24h")

        for h in agg.get("latest_headlines", [])[:5]:
            sent = h.get("sentiment")
            badge = "🟢" if (sent and sent > 0.1) else ("🔴" if (sent and sent < -0.1) else "🟡")
            st.markdown(
                f"{badge} **{h.get('title', '')}**  \n"
                f"<small>{h.get('source', '')} · {h.get('published_ts', '')[:16]}</small>",
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.info(f"News unavailable: {e}")


# ── Scoreboard ────────────────────────────────────────────────────────────────

def render_scoreboard(mode: str = "live") -> None:
    try:
        from src.scoreboard import get_scoreboard, resolve_pending_predictions

        resolve_pending_predictions(mode=mode.lower())
        sb = get_scoreboard(mode=mode.lower())

        col1, col2, col3 = st.columns(3)
        col1.metric("Total predictions", sb["total_predictions"])
        col2.metric(
            "Overall accuracy",
            f"{sb['overall_accuracy']:.1%}" if sb["overall_accuracy"] is not None else "N/A",
        )
        col3.metric(
            "Confident accuracy",
            f"{sb['confident_accuracy']:.1%}" if sb["confident_accuracy"] is not None else "N/A",
            help=f"Accuracy for predictions above {CONFIDENCE_THRESHOLD:.0%} confidence",
        )

        recent = sb.get("recent", [])
        if recent:
            st.markdown("**Recent resolved predictions:**")
            df = pd.DataFrame(recent)
            if "correct" in df.columns:
                df["✓"] = df["correct"].apply(lambda x: "✅" if x == 1 else "❌")
            st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.info(f"Scoreboard unavailable: {e}")


# ── Stock selector fallback ───────────────────────────────────────────────────

def ticker_selector_fallback(tickers: list[str]) -> str:
    display = [t.replace(".NS", "") for t in tickers]
    choice = st.selectbox("Or select any stock:", display, index=0, key="ticker_fallback")
    idx = display.index(choice)
    return tickers[idx]


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("📈 NSE Intraday Stock Direction Predictor")
    st.caption(DISCLAIMER)

    cfg = render_sidebar()
    mode = cfg["mode"]
    universe_name = cfg["universe"]

    # Market status banner
    if not _is_market_open():
        st.warning(
            "🔴 Market is currently **closed**. "
            "Showing data from the most recent session. "
            "Use **Replay** mode to simulate live predictions."
        )

    # Resolve universe
    try:
        from src.universe import get_nifty50, get_nifty100

        tickers_all = (
            get_nifty50() if universe_name == "NIFTY 50" else get_nifty100()
        )
        if not tickers_all:
            st.error("No validated tickers found. Run `python -m src.fetch_prices --validate` first.")
            return
    except Exception as e:
        st.error(f"Failed to load universe: {e}")
        return

    # ── Replay mode: cursor ────────────────────────────────────────────────────
    if mode == "Replay":
        if "replay_cursor" not in st.session_state:
            from src.replay import ReplayCursor
            st.session_state["replay_cursor"] = ReplayCursor(tickers_all[:10], n_days=5)

        cursor = st.session_state["replay_cursor"]

        replay_cfg = cfg.get("replay", {})
        if replay_cfg.get("next_bar"):
            cursor.advance()
            st.session_state["replay_cursor"] = cursor

        replay_as_of = cursor.current
        if replay_as_of:
            st.info(f"🎬 Replay cursor: **{replay_as_of.strftime('%Y-%m-%d %H:%M:%S IST')}**")
            st.progress(cursor.progress)
        else:
            st.error("No replay data available. Run fetch_prices first.")
            return
    else:
        replay_as_of = None

    # ── Load universe prices ──────────────────────────────────────────────────
    with st.spinner("Loading market data…"):
        prices = _fetch_universe_prices(tuple(tickers_all))

    if not prices:
        st.error("Could not load any price data. Check your internet connection.")
        return

    # ── Movers table ──────────────────────────────────────────────────────────
    st.subheader("📊 Top Movers Today")
    selected = render_movers_table(prices)

    # Fallback selectbox
    st.markdown("---")
    fallback_ticker = ticker_selector_fallback(tickers_all)
    chosen_ticker = selected or fallback_ticker

    # ── Stock detail panel ────────────────────────────────────────────────────
    st.subheader(f"🔍 {chosen_ticker.replace('.NS', '')}")

    # Candlestick
    with st.expander("Candlestick Chart", expanded=True):
        df_chart = prices.get(chosen_ticker, pd.DataFrame())
        render_candlestick(chosen_ticker, df_chart)

    # Prediction
    with st.expander("🤖 Direction Prediction (15 min)", expanded=True):
        try:
            from src.predict import predict_for
            from src.scoreboard import log_prediction

            predict_mode = "replay" if mode == "Replay" else "live"
            with st.spinner("Running model…"):
                result = predict_for(
                    chosen_ticker,
                    as_of=replay_as_of,
                    mode=predict_mode,
                    include_news=cfg.get("include_news", True),
                    news_weight=cfg.get("news_weight", 0.35),
                )

            render_prediction(result)
            log_prediction(result)

        except FileNotFoundError:
            st.error(
                "Model not found. Run `python -m src.train` to train the model first."
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # News & Sentiment
    with st.expander("📰 News & Sentiment", expanded=False):
        render_news(chosen_ticker)

    # Scoreboard
    with st.expander("🏆 Prediction Scoreboard", expanded=False):
        render_scoreboard(mode=mode)

    st.markdown("---")
    st.caption(
        "Data: yfinance (Yahoo Finance) | Sentiment: ProsusAI/FinBERT | "
        "Index: NSE NIFTY 50 / NIFTY 100 | "
        f"Last updated: {datetime.now(IST).strftime('%H:%M:%S IST')}"
    )


if __name__ == "__main__":
    main()
