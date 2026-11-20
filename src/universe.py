"""
universe.py – NIFTY 100 ticker list, validation, and company name mapping.

Validation tries a minimal yfinance download for each ticker and drops
any that fail (renamed, delisted, or wrong symbol).  The validated list is
cached to data/validated_tickers.json so subsequent runs skip the check.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── Raw NIFTY 100 constituent list (NSE tickers with .NS suffix) ──────────────
# Source: NSE India NIFTY 100 index (approximate, as of late 2024).
# A handful of symbols may need updating if NSE renames them.
NIFTY_100_RAW: list[str] = [
    # NIFTY 50
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ICICIBANK.NS",
    "INFY.NS", "SBIN.NS", "HINDUNILVR.NS", "ITC.NS", "LT.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "NTPC.NS", "POWERGRID.NS", "ULTRACEMCO.NS",
    "WIPRO.NS", "HCLTECH.NS", "ONGC.NS", "COALINDIA.NS", "NESTLEIND.NS",
    "BAJAJFINSV.NS", "JSWSTEEL.NS", "TATAMOTORS.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "HINDALCO.NS", "TATACONSUM.NS", "DRREDDY.NS", "CIPLA.NS", "BPCL.NS",
    "BRITANNIA.NS", "INDUSINDBK.NS", "APOLLOHOSP.NS", "DIVISLAB.NS", "EICHERMOT.NS",
    "GRASIM.NS", "HEROMOTOCO.NS", "SHRIRAMFIN.NS", "SBILIFE.NS", "HDFCLIFE.NS",
    "M&M.NS", "TATASTEEL.NS", "BAJAJ-AUTO.NS", "TECHM.NS", "LTIM.NS",
    # Additional NIFTY 100 (non-NIFTY 50)
    "ABB.NS", "ADANIGREEN.NS", "ATGL.NS", "AMBUJACEM.NS", "BANKBARODA.NS",
    "BERGEPAINT.NS", "BEL.NS", "BOSCHLTD.NS", "CANBK.NS", "CHOLAFIN.NS",
    "COLPAL.NS", "DABUR.NS", "DLF.NS", "GAIL.NS", "GODREJCP.NS",
    "HAVELLS.NS", "ICICIPRULI.NS", "ICICIGI.NS", "IOC.NS", "IRCTC.NS",
    "JINDALSTEL.NS", "JIOFIN.NS", "JUBLFOOD.NS", "LICI.NS", "LODHA.NS",
    "LUPIN.NS", "MARICO.NS", "MOTHERSON.NS", "MUTHOOTFIN.NS", "NAUKRI.NS",
    "NHPC.NS", "NMDC.NS", "OFSS.NS", "PAGEIND.NS", "PERSISTENT.NS",
    "PIIND.NS", "PIDILITIND.NS", "PNB.NS", "RECLTD.NS", "SAIL.NS",
    "SBICARD.NS", "SIEMENS.NS", "TATAPOWER.NS", "TORNTPHARM.NS", "TRENT.NS",
    "TVSMOTOR.NS", "UBL.NS", "UNIONBANK.NS", "VBL.NS", "VEDL.NS",
]

# NIFTY 50 subset (first 50 entries above)
NIFTY_50_RAW: list[str] = NIFTY_100_RAW[:50]

# ── Company name mapping (for Google News RSS search) ────────────────────────
TICKER_TO_NAME: dict[str, str] = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "BHARTIARTL.NS": "Bharti Airtel",
    "ICICIBANK.NS": "ICICI Bank",
    "INFY.NS": "Infosys",
    "SBIN.NS": "State Bank of India",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC Limited",
    "LT.NS": "Larsen Toubro",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "AXISBANK.NS": "Axis Bank",
    "BAJFINANCE.NS": "Bajaj Finance",
    "ASIANPAINT.NS": "Asian Paints",
    "MARUTI.NS": "Maruti Suzuki",
    "SUNPHARMA.NS": "Sun Pharmaceutical",
    "TITAN.NS": "Titan Company",
    "NTPC.NS": "NTPC Limited",
    "POWERGRID.NS": "Power Grid Corporation",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "WIPRO.NS": "Wipro",
    "HCLTECH.NS": "HCL Technologies",
    "ONGC.NS": "Oil and Natural Gas Corporation",
    "COALINDIA.NS": "Coal India",
    "NESTLEIND.NS": "Nestle India",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "JSWSTEEL.NS": "JSW Steel",
    "TATAMOTORS.NS": "Tata Motors",
    "ADANIENT.NS": "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports",
    "HINDALCO.NS": "Hindalco Industries",
    "TATACONSUM.NS": "Tata Consumer Products",
    "DRREDDY.NS": "Dr Reddys Laboratories",
    "CIPLA.NS": "Cipla",
    "BPCL.NS": "Bharat Petroleum",
    "BRITANNIA.NS": "Britannia Industries",
    "INDUSINDBK.NS": "IndusInd Bank",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "DIVISLAB.NS": "Divis Laboratories",
    "EICHERMOT.NS": "Eicher Motors",
    "GRASIM.NS": "Grasim Industries",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "SHRIRAMFIN.NS": "Shriram Finance",
    "SBILIFE.NS": "SBI Life Insurance",
    "HDFCLIFE.NS": "HDFC Life Insurance",
    "M&M.NS": "Mahindra and Mahindra",
    "TATASTEEL.NS": "Tata Steel",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "TECHM.NS": "Tech Mahindra",
    "LTIM.NS": "LTIMindtree",
    "ABB.NS": "ABB India",
    "ADANIGREEN.NS": "Adani Green Energy",
    "ATGL.NS": "Adani Total Gas",
    "AMBUJACEM.NS": "Ambuja Cements",
    "BANKBARODA.NS": "Bank of Baroda",
    "BERGEPAINT.NS": "Berger Paints",
    "BEL.NS": "Bharat Electronics",
    "BOSCHLTD.NS": "Bosch India",
    "CANBK.NS": "Canara Bank",
    "CHOLAFIN.NS": "Cholamandalam Investment",
    "COLPAL.NS": "Colgate Palmolive India",
    "DABUR.NS": "Dabur India",
    "DLF.NS": "DLF Limited",
    "GAIL.NS": "GAIL India",
    "GODREJCP.NS": "Godrej Consumer Products",
    "HAVELLS.NS": "Havells India",
    "ICICIPRULI.NS": "ICICI Prudential Life",
    "ICICIGI.NS": "ICICI Lombard General Insurance",
    "IOC.NS": "Indian Oil Corporation",
    "IRCTC.NS": "Indian Railway Catering",
    "JINDALSTEL.NS": "Jindal Steel and Power",
    "JIOFIN.NS": "Jio Financial Services",
    "JUBLFOOD.NS": "Jubilant FoodWorks",
    "LICI.NS": "Life Insurance Corporation",
    "LODHA.NS": "Macrotech Developers",
    "LUPIN.NS": "Lupin Limited",
    "MARICO.NS": "Marico Limited",
    "MOTHERSON.NS": "Samvardhana Motherson",
    "MUTHOOTFIN.NS": "Muthoot Finance",
    "NAUKRI.NS": "Info Edge India",
    "NHPC.NS": "NHPC Limited",
    "NMDC.NS": "NMDC Limited",
    "OFSS.NS": "Oracle Financial Services",
    "PAGEIND.NS": "Page Industries",
    "PERSISTENT.NS": "Persistent Systems",
    "PIIND.NS": "PI Industries",
    "PIDILITIND.NS": "Pidilite Industries",
    "PNB.NS": "Punjab National Bank",
    "RECLTD.NS": "REC Limited",
    "SAIL.NS": "Steel Authority of India",
    "SBICARD.NS": "SBI Cards and Payment Services",
    "SIEMENS.NS": "Siemens India",
    "TATAPOWER.NS": "Tata Power",
    "TORNTPHARM.NS": "Torrent Pharmaceuticals",
    "TRENT.NS": "Trent Limited",
    "TVSMOTOR.NS": "TVS Motor Company",
    "UBL.NS": "United Breweries",
    "UNIONBANK.NS": "Union Bank of India",
    "VBL.NS": "Varun Beverages",
    "VEDL.NS": "Vedanta Limited",
}

_VALIDATED_CACHE_FILE: Optional[str] = None


def _get_cache_path() -> str:
    """Return path for validated ticker cache."""
    from config import DATA_DIR
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, "validated_tickers.json")


def validate_tickers(
    tickers: list[str],
    force: bool = False,
    period: str = "5d",
) -> list[str]:
    """
    Validate tickers by attempting a minimal yfinance download.

    Parameters
    ----------
    tickers : list of raw ticker strings
    force   : re-validate even if cache exists
    period  : yfinance period string for the test download

    Returns
    -------
    List of tickers that returned at least one row of data.
    """
    import yfinance as yf

    cache_path = _get_cache_path()

    if not force and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        logger.info("Loaded %d validated tickers from cache.", len(cached))
        return cached

    logger.info("Validating %d tickers (this may take a few minutes)…", len(tickers))
    valid: list[str] = []
    failed: list[str] = []

    from config import DOWNLOAD_BATCH_SIZE, DOWNLOAD_SLEEP_SEC

    for i in range(0, len(tickers), DOWNLOAD_BATCH_SIZE):
        batch = tickers[i : i + DOWNLOAD_BATCH_SIZE]
        try:
            data = yf.download(
                batch,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            for t in batch:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        if t in data.columns.get_level_values(0):
                            df = data[t]
                        elif t in data.columns.get_level_values(1):
                            df = data.xs(t, level=1, axis=1)
                        else:
                            df = data if len(batch) == 1 else None
                    else:
                        df = data
                    if df is not None and not df.empty and len(df) > 0:
                        valid.append(t)
                    else:
                        failed.append(t)
                        logger.warning("Ticker %s returned no data — dropping.", t)
                except Exception:
                    failed.append(t)
                    logger.warning("Ticker %s caused an error — dropping.", t)
        except Exception as e:
            logger.error("Batch %s failed: %s — marking all as failed.", batch, e)
            failed.extend(batch)

        if i + DOWNLOAD_BATCH_SIZE < len(tickers):
            time.sleep(DOWNLOAD_SLEEP_SEC)

    logger.info(
        "Validation complete: %d valid, %d failed. Failed: %s",
        len(valid), len(failed), failed,
    )

    with open(cache_path, "w") as f:
        json.dump(valid, f, indent=2)

    return valid


def get_nifty100(force_validate: bool = False) -> list[str]:
    """Return validated NIFTY 100 ticker list."""
    return validate_tickers(NIFTY_100_RAW, force=force_validate)


def get_nifty50(force_validate: bool = False) -> list[str]:
    """Return validated NIFTY 50 ticker list."""
    all_valid = set(get_nifty100(force_validate=force_validate))
    return [t for t in NIFTY_50_RAW if t in all_valid]


def ticker_to_company(ticker: str) -> str:
    """Return the human-readable company name for news search."""
    return TICKER_TO_NAME.get(ticker, ticker.replace(".NS", ""))
