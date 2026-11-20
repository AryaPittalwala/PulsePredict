"""
evaluate.py – Walk-forward + held-out-ticker evaluation, baselines, and
              cost-aware backtest.

Baselines
---------
1. Majority class          – always predict the most common class
2. Momentum                – predict direction of the last 1-bar return
3. Always NEUTRAL

Metrics
-------
- Accuracy, macro-F1, per-class P/R, confusion matrix
- Log loss, calibration check
- Directional accuracy above CONFIDENCE_THRESHOLD
- Cost-aware backtest (long on UP, short on DOWN, subtract ROUND_TRIP_COST)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

sys.path.insert(0, ".")

from config import (
    CLASS_DOWN,
    CLASS_MAP,
    CLASS_NAMES,
    CLASS_NEUTRAL,
    CLASS_UP,
    CONFIDENCE_THRESHOLD,
    RESULTS_DIR,
    ROUND_TRIP_COST,
)

logger = logging.getLogger(__name__)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _int_to_label(arr: np.ndarray) -> list[str]:
    inv = {v: k for k, v in CLASS_MAP.items()}
    return [inv[i] for i in arr]


# ── Baselines ─────────────────────────────────────────────────────────────────

def baseline_majority(y_true: pd.Series) -> np.ndarray:
    majority = y_true.value_counts().idxmax()
    return np.full(len(y_true), majority)


def baseline_momentum(df: pd.DataFrame) -> np.ndarray:
    """Predict direction of the last 1-bar return (ret_lag1)."""
    if "ret_lag1" not in df.columns:
        return np.full(len(df), CLASS_NEUTRAL)
    preds = np.where(
        df["ret_lag1"] > 0, CLASS_UP, np.where(df["ret_lag1"] < 0, CLASS_DOWN, CLASS_NEUTRAL)
    )
    return preds


def baseline_always_neutral(y_true: pd.Series) -> np.ndarray:
    return np.full(len(y_true), CLASS_NEUTRAL)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_split(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    split_name: str = "test",
) -> dict:
    """
    Evaluate a trained model on a labelled DataFrame.

    Returns a dict with all metrics.
    """
    if df.empty or "label" not in df.columns:
        logger.warning("Empty or unlabelled split: %s", split_name)
        return {}

    present_cols = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(present_cols)
    if missing:
        logger.warning("Missing feature columns in %s: %s", split_name, missing)

    X = df[present_cols].values
    y_true_int = df["label_int"].values.astype(int)
    y_true_str = df["label"].values

    # Model predictions
    proba = model.predict_proba(X)
    y_pred_int = np.argmax(proba, axis=1)
    y_pred_str = np.array(_int_to_label(y_pred_int))
    confidence = np.max(proba, axis=1)

    # ── Classification metrics ────────────────────────────────────────────────
    acc = accuracy_score(y_true_str, y_pred_str)
    macro_f1 = f1_score(y_true_str, y_pred_str, average="macro", zero_division=0)
    report = classification_report(
        y_true_str, y_pred_str, target_names=CLASS_NAMES, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true_str, y_pred_str, labels=CLASS_NAMES)

    # ── Log loss ──────────────────────────────────────────────────────────────
    try:
        ll = log_loss(y_true_int, proba)
    except Exception:
        ll = float("nan")

    # ── Directional accuracy above threshold ──────────────────────────────────
    high_conf_mask = (confidence >= CONFIDENCE_THRESHOLD) & (
        (y_pred_str == CLASS_UP) | (y_pred_str == CLASS_DOWN)
    )
    if high_conf_mask.sum() > 0:
        # Directional: sign of actual fwd_ret matches predicted direction
        if "fwd_ret" in df.columns:
            actual_fwd = df["fwd_ret"].values
            pred_dir = np.where(y_pred_str == CLASS_UP, 1, -1)
            actual_sign = np.sign(actual_fwd)
            correct_dir = (pred_dir == actual_sign)[high_conf_mask]
            dir_acc = correct_dir.mean()
        else:
            dir_acc = float("nan")
    else:
        dir_acc = float("nan")

    n_high_conf = int(high_conf_mask.sum())

    # ── Baselines ─────────────────────────────────────────────────────────────
    y_majority = baseline_majority(df["label"])
    y_momentum = baseline_momentum(df)
    y_neutral = baseline_always_neutral(df["label"])

    baseline_results = {
        "majority": {
            "acc": accuracy_score(y_true_str, y_majority),
            "macro_f1": f1_score(y_true_str, y_majority, average="macro", zero_division=0),
        },
        "momentum": {
            "acc": accuracy_score(y_true_str, y_momentum),
            "macro_f1": f1_score(y_true_str, y_momentum, average="macro", zero_division=0),
        },
        "always_neutral": {
            "acc": accuracy_score(y_true_str, y_neutral),
            "macro_f1": f1_score(y_true_str, y_neutral, average="macro", zero_division=0),
        },
    }

    # ── Cost-aware backtest ───────────────────────────────────────────────────
    backtest = run_backtest(df, proba, y_pred_str, confidence)

    result = {
        "split": split_name,
        "n_rows": len(df),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "log_loss": float(ll),
        "directional_accuracy": float(dir_acc) if not np.isnan(dir_acc) else None,
        "n_high_confidence": n_high_conf,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "baselines": baseline_results,
        "backtest": backtest,
    }

    return result


def run_backtest(
    df: pd.DataFrame,
    proba: np.ndarray,
    y_pred_str: np.ndarray,
    confidence: np.ndarray,
) -> dict:
    """
    Simple cost-aware backtest.

    Rules
    -----
    - Enter long on high-confidence UP, short on high-confidence DOWN.
    - Exit after HORIZON_BARS bars.
    - No overlapping positions per ticker.
    - Subtract ROUND_TRIP_COST per trade.
    """
    from config import HORIZON_BARS

    if "fwd_ret" not in df.columns:
        return {}

    act_mask = (confidence >= CONFIDENCE_THRESHOLD) & (
        (y_pred_str == CLASS_UP) | (y_pred_str == CLASS_DOWN)
    )
    if act_mask.sum() == 0:
        return {"n_trades": 0}

    sub = df.copy()
    sub["_pred"] = y_pred_str
    sub["_conf"] = confidence
    sub["_act"] = act_mask

    trades = []
    tickers = sub["ticker"].unique() if "ticker" in sub.columns else [None]

    for ticker in tickers:
        t_sub = sub[sub["ticker"] == ticker] if ticker else sub
        last_exit_idx = -HORIZON_BARS - 1

        for i, (idx, row) in enumerate(t_sub.iterrows()):
            if not row["_act"]:
                continue
            if i <= last_exit_idx + HORIZON_BARS:
                continue  # no overlapping positions

            direction = 1 if row["_pred"] == CLASS_UP else -1
            ret = row["fwd_ret"] * direction - ROUND_TRIP_COST
            trades.append(ret)
            last_exit_idx = i

    if not trades:
        return {"n_trades": 0}

    trades_arr = np.array(trades)
    cum_ret = np.cumsum(trades_arr)
    drawdown = cum_ret - np.maximum.accumulate(cum_ret)
    max_dd = float(drawdown.min())
    sharpe = (
        float(trades_arr.mean() / trades_arr.std() * np.sqrt(252 * 75))
        if trades_arr.std() > 0 else float("nan")
    )

    return {
        "n_trades": len(trades),
        "hit_rate": float((trades_arr > 0).mean()),
        "net_ret_per_trade": float(trades_arr.mean()),
        "cumulative_ret": float(trades_arr.sum()),
        "max_drawdown": max_dd,
        "sharpe_like": sharpe,
        "note": (
            "Optimistic backtest. Ignores market impact, assumes instant fill "
            "at next bar open, no position sizing."
        ),
    }


def print_report(report: dict, title: str) -> None:
    """Pretty-print an evaluation report to stdout."""
    if not report:
        print(f"\n[{title}] — no data")
        return

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Rows          : {report.get('n_rows')}")
    print(f"  Accuracy      : {report.get('accuracy', 0):.4f}")
    print(f"  Macro-F1      : {report.get('macro_f1', 0):.4f}")
    print(f"  Log Loss      : {report.get('log_loss', 0):.4f}")
    dir_acc = report.get("directional_accuracy")
    print(f"  Dir. Accuracy : {dir_acc:.4f}" if dir_acc else "  Dir. Accuracy : n/a")
    print(f"  High-conf bars: {report.get('n_high_confidence')}")

    b = report.get("baselines", {})
    if b:
        print(f"\n  Baselines:")
        for name, m in b.items():
            print(f"    {name:<18} acc={m['acc']:.4f}  macro-F1={m['macro_f1']:.4f}")

    bt = report.get("backtest", {})
    if bt and bt.get("n_trades", 0) > 0:
        print(f"\n  Backtest (cost-adjusted):")
        print(f"    Trades        : {bt['n_trades']}")
        print(f"    Hit rate      : {bt.get('hit_rate', 0):.4f}")
        print(f"    Net ret/trade : {bt.get('net_ret_per_trade', 0):.5f}")
        print(f"    Cumul. return : {bt.get('cumulative_ret', 0):.4f}")
        print(f"    Max drawdown  : {bt.get('max_drawdown', 0):.4f}")
        print(f"    Sharpe-like   : {bt.get('sharpe_like', float('nan')):.2f}")

    print()


def save_results(report: dict, split_name: str) -> None:
    """Save evaluation report as JSON to results/."""
    import json
    path = os.path.join(RESULTS_DIR, f"{split_name}_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Results saved to %s", path)


def write_summary(reports: list[dict], out_path: Optional[str] = None) -> None:
    """Write a human-readable results/summary.md."""
    out_path = out_path or os.path.join(RESULTS_DIR, "summary.md")
    lines = [
        "# Evaluation Summary\n",
        "> **Educational project — not financial advice.**\n",
        "---\n",
    ]
    for r in reports:
        if not r:
            continue
        title = r.get("split", "unknown")
        lines.append(f"## {title}\n")
        lines.append(f"- Rows: {r.get('n_rows')}\n")
        lines.append(f"- Accuracy: {r.get('accuracy', 0):.4f}\n")
        lines.append(f"- Macro-F1: {r.get('macro_f1', 0):.4f}\n")
        da = r.get("directional_accuracy")
        lines.append(f"- Directional accuracy (above threshold): {da:.4f}\n" if da else "- Directional accuracy: n/a\n")
        b = r.get("baselines", {})
        if b:
            lines.append("\n### Baselines\n")
            for name, m in b.items():
                lines.append(f"- {name}: acc={m['acc']:.4f}, macro-F1={m['macro_f1']:.4f}\n")
        bt = r.get("backtest", {})
        if bt and bt.get("n_trades", 0) > 0:
            lines.append("\n### Cost-adjusted backtest\n")
            lines.append(f"- Trades: {bt['n_trades']}\n")
            lines.append(f"- Hit rate: {bt.get('hit_rate', 0):.4f}\n")
            lines.append(f"- Cumul. return: {bt.get('cumulative_ret', 0):.4f}\n")
            lines.append(f"- Max drawdown: {bt.get('max_drawdown', 0):.4f}\n")
            lines.append(f"\n*{bt.get('note', '')}*\n")
        lines.append("\n---\n")

    lines.append(
        "\n## Honest assessment\n"
        "Short-horizon intraday prediction is extremely difficult. NSE markets "
        "are approximately efficient at the 15-minute scale. If the model does "
        "not beat baselines by a statistically meaningful margin, that is the "
        "expected result. This project is about engineering and rigorous evaluation, "
        "not guaranteed profitability.\n"
    )

    with open(out_path, "w") as f:
        f.writelines(lines)
    logger.info("Summary written to %s", out_path)
