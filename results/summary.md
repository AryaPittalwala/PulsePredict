# Evaluation Summary
> **Educational project — not financial advice.**
---
## test_seen
- Rows: 9471
- Accuracy: 0.3905
- Macro-F1: 0.3827
- Directional accuracy: n/a

### Baselines
- majority: acc=0.3458, macro-F1=0.1713
- momentum: acc=0.3308, macro-F1=0.2828
- always_neutral: acc=0.3458, macro-F1=0.1713

---
## test_held_out
- Rows: 2375
- Accuracy: 0.4392
- Macro-F1: 0.4276
- Directional accuracy: n/a

### Baselines
- majority: acc=0.3566, macro-F1=0.1753
- momentum: acc=0.3196, macro-F1=0.2735
- always_neutral: acc=0.3566, macro-F1=0.1753

---

## Honest assessment
Short-horizon intraday prediction is extremely difficult. NSE markets are approximately efficient at the 15-minute scale. If the model does not beat baselines by a statistically meaningful margin, that is the expected result. This project is about engineering and rigorous evaluation, not guaranteed profitability.
