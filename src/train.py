"""
train.py – Pooled LightGBM multiclass training with walk-forward splits
           and held-out-ticker generalization evaluation.

Training design
---------------
1.  Hold out HELDOUT_TICKER_FRACTION of tickers (seeded random choice).
    These are NEVER in training.
2.  On the remaining tickers, split time-wise:
      train  : earliest portion
      val    : middle portion (with HORIZON_BARS purge gap from train)
      test   : latest portion (with purge gap from val)
3.  Train LightGBM with class weights and early stopping on val.
4.  Evaluate on (a) test split seen tickers, (b) held-out tickers on
    the same test time window.
5.  Save model + metadata.json.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")

from config import (
    CLASS_MAP,
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    HELDOUT_TICKER_FRACTION,
    HORIZON_BARS,
    INDEX_TICKER,
    METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    TZ,
)
from src.db import init_db, query_prices_multi, list_tickers
from src.features import build_features, get_feature_columns
from src.labels import add_labels_to_features, compute_class_weights
from src.universe import get_nifty100

logger = logging.getLogger(__name__)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_data(tickers: list[str]) -> pd.DataFrame:
    """
    Load, feature-engineer, and label data for all training tickers.

    Returns a DataFrame with feature columns + label columns, indexed by
    (ticker, ts).
    """
    # Load Nifty index for market context
    nifty_raw = query_prices_multi([INDEX_TICKER])
    if nifty_raw.empty:
        logger.warning("No Nifty index data found. Market context features will be NaN.")
        nifty_df: Optional[pd.DataFrame] = None
    else:
        nifty_df = nifty_raw.copy()
        nifty_df["ts"] = pd.to_datetime(nifty_df["ts"], utc=False)
        nifty_df = nifty_df.set_index("ts").sort_index()
        if nifty_df.index.tzinfo is None:
            nifty_df.index = nifty_df.index.tz_localize(TZ)

    all_dfs: list[pd.DataFrame] = []

    for ticker in tickers:
        raw = query_prices_multi([ticker])
        if raw.empty or len(raw) < 100:
            logger.warning("Skipping %s: too few rows (%d).", ticker, len(raw))
            continue

        ohlcv = raw.copy()
        ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=False)
        ohlcv = ohlcv.set_index("ts").sort_index()
        if ohlcv.index.tzinfo is None:
            ohlcv.index = ohlcv.index.tz_localize(TZ)

        try:
            feat_df = build_features(ohlcv, nifty_df=nifty_df, drop_nans=True)
            labelled = add_labels_to_features(feat_df, ohlcv)
            labelled["ticker"] = ticker
            all_dfs.append(labelled)
        except Exception as e:
            logger.error("Feature/label error for %s: %s", ticker, e)
            continue

    if not all_dfs:
        raise RuntimeError("No data available after feature/label pipeline.")

    combined = pd.concat(all_dfs, axis=0).sort_index()
    logger.info("Combined dataset: %d rows, %d tickers.", len(combined), len(all_dfs))
    return combined


# ── Splits ────────────────────────────────────────────────────────────────────

def split_held_out(
    tickers: list[str],
    fraction: float = HELDOUT_TICKER_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[list[str], list[str]]:
    """
    Randomly split tickers into training and held-out sets.

    Returns (train_tickers, held_out_tickers).
    """
    rng = np.random.default_rng(seed)
    n_held = max(1, int(len(tickers) * fraction))
    held_idx = rng.choice(len(tickers), size=n_held, replace=False)
    held_out = [tickers[i] for i in sorted(held_idx)]
    train = [t for t in tickers if t not in set(held_out)]
    logger.info(
        "Held-out tickers (%d): %s", len(held_out), held_out[:10]
    )
    return train, held_out


def time_split(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    purge_bars: int = HORIZON_BARS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-ordered DataFrame into train/val/test with purge gaps.

    The purge gap prevents the horizon-length label overlap between splits.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end]
    val = df.iloc[train_end + purge_bars : val_end]
    test = df.iloc[val_end + purge_bars :]

    logger.info(
        "Time split — train: %d, val: %d, test: %d (purge=%d bars).",
        len(train), len(val), len(test), purge_bars,
    )
    return train, val, test


# ── Model training ────────────────────────────────────────────────────────────

def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> LGBMClassifier:
    """
    Train a LightGBM multiclass classifier with class weights and early stopping.
    """
    X_train = train_df[feature_cols].values
    y_train = train_df["label_int"].values.astype(int)
    X_val = val_df[feature_cols].values
    y_val = val_df["label_int"].values.astype(int)

    # Class weights computed from training labels only
    cw_dict = compute_class_weights(train_df["label"])
    class_weight_arr = np.array([cw_dict[i] for i in range(len(CLASS_NAMES))])

    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight={i: cw_dict[i] for i in range(len(CLASS_NAMES))},
        objective="multiclass",
        num_class=3,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(period=-1),
        ],
    )
    logger.info("Model trained. Best iteration: %d", model.best_iteration_)
    return model


# ── Saving ────────────────────────────────────────────────────────────────────

def save_model(
    model: LGBMClassifier,
    feature_cols: list[str],
    train_df: pd.DataFrame,
    test_metrics: dict,
) -> None:
    """Persist model and metadata."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", MODEL_PATH)

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "feature_columns": feature_cols,
        "class_names": CLASS_NAMES,
        "class_map": CLASS_MAP,
        "train_start": str(train_df.index.min()),
        "train_end": str(train_df.index.max()),
        "n_train_rows": int(len(train_df)),
        "best_iteration": int(model.best_iteration_),
        "test_metrics": test_metrics,
        "config": {
            "HORIZON_BARS": HORIZON_BARS,
            "RANDOM_SEED": RANDOM_SEED,
            "HELDOUT_TICKER_FRACTION": HELDOUT_TICKER_FRACTION,
            "CONFIDENCE_THRESHOLD": CONFIDENCE_THRESHOLD,
        },
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Metadata saved to %s", METADATA_PATH)


def load_model() -> tuple[LGBMClassifier, dict]:
    """Load saved model and metadata.json."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return model, metadata


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    init_db()
    all_tickers = [t for t in list_tickers() if t != INDEX_TICKER]
    logger.info("Found %d non-index tickers in DB.", len(all_tickers))

    if len(all_tickers) < 5:
        raise RuntimeError("Not enough tickers in DB. Run fetch_prices first.")

    train_tickers, held_out_tickers = split_held_out(all_tickers)
    feature_cols = get_feature_columns()

    # ── Load & split training tickers ─────────────────────────────────────────
    logger.info("Loading data for %d training tickers…", len(train_tickers))
    train_data = load_all_data(train_tickers)
    train_df, val_df, test_df = time_split(train_data)

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Training model on %d rows…", len(train_df))
    model = train_model(train_df, val_df, feature_cols)

    # ── Evaluate on seen-ticker test set ──────────────────────────────────────
    from src.evaluate import evaluate_split, print_report, save_results, write_summary

    test_report = evaluate_split(model, test_df, feature_cols, split_name="test_seen")
    print_report(test_report, "Test (seen tickers)")
    save_results(test_report, "test_seen")

    # ── Evaluate on held-out tickers ──────────────────────────────────────────
    held_report = {}
    if held_out_tickers:
        logger.info("Loading held-out ticker data…")
        held_data = load_all_data(held_out_tickers)
        # Use the same test time window as the seen-ticker test
        test_start = test_df.index.min()
        test_end = test_df.index.max()
        held_test = held_data[
            (held_data.index >= test_start) & (held_data.index <= test_end)
        ]
        if len(held_test) > 0:
            held_report = evaluate_split(
                model, held_test, feature_cols, split_name="test_held_out"
            )
            print_report(held_report, "Test (held-out tickers)")
            save_results(held_report, "test_held_out")
        else:
            logger.warning("No held-out test data in the test time window.")

    write_summary([test_report, held_report])

    test_metrics = {
        "seen_accuracy": test_report.get("accuracy"),
        "seen_macro_f1": test_report.get("macro_f1"),
        "held_accuracy": held_report.get("accuracy") if held_out_tickers else None,
    }
    save_model(model, feature_cols, train_df, test_metrics)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
