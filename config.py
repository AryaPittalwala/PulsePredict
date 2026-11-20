"""
Centralized configuration for NSE Intraday Stock Direction Predictor.
All constants live here — no magic numbers elsewhere in the codebase.
"""

# ── Patch curl_cffi on Windows to bypass missing CA root certs ────────────────
try:
    from curl_cffi.requests.session import Session as _CurlSession
    _orig_curl_init = _CurlSession.__init__
    def _patched_curl_init(self, *args, **kwargs):
        kwargs["verify"] = False
        _orig_curl_init(self, *args, **kwargs)
    _CurlSession.__init__ = _patched_curl_init
except Exception:
    pass

# ── Timezone & market hours ──────────────────────────────────────────────────
TZ = "Asia/Kolkata"
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"

# ── Data ─────────────────────────────────────────────────────────────────────
INTERVAL = "5m"
INDEX_TICKER = "^NSEI"

# ── Model target ─────────────────────────────────────────────────────────────
HORIZON_BARS = 3           # 3 × 5 min = 15-minute lookahead
NEUTRAL_K = 0.3            # neutral band = NEUTRAL_K × vol × sqrt(HORIZON_BARS)
VOL_WINDOW = 36            # bars for rolling vol used in label threshold

# ── Training / evaluation ─────────────────────────────────────────────────────
HELDOUT_TICKER_FRACTION = 0.2   # fraction of tickers withheld from training
ROUND_TRIP_COST = 0.001         # 0.1% per round-trip (brokerage + taxes + slippage)
CONFIDENCE_THRESHOLD = 0.55     # min model confidence to "act" in backtest
RANDOM_SEED = 42

# ── Feature engineering ───────────────────────────────────────────────────────
RETURN_LAGS = [1, 2, 3, 6, 12]       # log-return lag lengths (bars)
VOL_WINDOWS = [12, 36]               # rolling-std windows for volatility features
RSI_PERIOD = 14
EMA_SHORT = 9
EMA_LONG = 21
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VOLUME_MA_BARS = 78                  # ~1 full trading session of 5-min bars

# ── Sentiment & News Integration ─────────────────────────────────────────────
USE_SENTIMENT_FEATURES = False
FINBERT_MODEL = "ProsusAI/finbert"
LIVE_NEWS_SENTIMENT_ENABLED = True      # adjust prediction with breaking live news
NEWS_SENTIMENT_WEIGHT = 0.35            # maximum influence of news sentiment on prediction
NEWS_SENTIMENT_WINDOW_HOURS = 24        # hours lookback for active breaking news

# ── Download / storage ────────────────────────────────────────────────────────
DOWNLOAD_BATCH_SIZE = 10     # tickers per yfinance request
DOWNLOAD_SLEEP_SEC = 2.0     # sleep between batches
DOWNLOAD_RETRIES = 3         # retry attempts per batch
FETCH_DAYS = 5               # incremental fetch window (days)
HISTORY_DAYS = 60            # max historical window yfinance serves for 5m

# ── Paths ─────────────────────────────────────────────────────────────────────
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

DB_PATH = os.path.join(DATA_DIR, "market.db")
MODEL_PATH = os.path.join(MODELS_DIR, "lgbm_model.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

# ── Labels ───────────────────────────────────────────────────────────────────
CLASS_UP = "UP"
CLASS_DOWN = "DOWN"
CLASS_NEUTRAL = "NEUTRAL"
CLASS_MAP = {CLASS_UP: 2, CLASS_DOWN: 0, CLASS_NEUTRAL: 1}
CLASS_NAMES = [CLASS_DOWN, CLASS_NEUTRAL, CLASS_UP]  # index → name
