# FraudGraph Evaluation Report

## Overview

This report presents offline evaluation metrics for fraud detection models on synthetic transaction data with labeled fraud patterns.

## Dataset Split

- **Train**: 48,486 transactions (2.25% fraud)
- **Validation**: 16,162 transactions (3.85% fraud)
- **Test**: 16,162 transactions (1.84% fraud)

Time-based split (oldest → newest) to prevent leakage.

## Models Evaluated

1. **Rules Baseline**: Hand-crafted thresholds (velocity, new device, high-risk MCC, amount spike)
2. **Logistic Regression (Base)**: Tabular features only
3. **Logistic Regression (Graph)**: Tabular + graph features
4. **LightGBM (Base)**: Tabular features only
5. **LightGBM (Graph)**: Tabular + graph features *(best)*

## Test Set Results

| Model | PR-AUC | ROC-AUC | P@50 | P@100 | P@200 |
|-------|--------|---------|------|-------|-------|
| Rules | 0.042 | 0.319 | 12.0% | 6.0% | 9.0% |
| LR (Base) | 0.655 | 0.965 | 92.0% | 84.0% | 75.5% |
| LR (Graph) | 0.837 | 0.998 | 88.0% | 88.0% | 80.5% |
| LGB (Base) | 0.854 | 0.985 | 100.0% | 100.0% | 95.5% |
| **LGB (Graph)** | **0.998** | **1.000** | **100.0%** | **100.0%** | **100.0%** |

## Key Findings

### 1. Graph Features Provide Strong Lift

- **LR**: Base (0.655 PR-AUC) → +Graph (0.837) = **+27.8% lift**
- **LightGBM**: Base (0.854 PR-AUC) → +Graph (0.998) = **+16.9% lift**

Graph features (shared devices, account/merchant degree, new merchant/device flags) capture relational fraud patterns effectively.

### 2. Precision@K Performance

At **K=200** (alert budget):
- Rules baseline: 9% precision (mostly noise)
- LightGBM + Graph: **100% precision** (all top 200 are fraud)

This means an analyst reviewing the top 200 alerts would find ~200 fraudulent transactions with near-zero false positives.

### 3. Model Comparison

- **Rules**: Poor performance; hard to tune thresholds for all patterns
- **Tabular-only models**: Good baseline (0.65-0.85 PR-AUC)
- **Graph-enhanced models**: Excellent performance (0.84-0.998 PR-AUC)
- **LightGBM > Logistic Regression**: Tree-based models capture non-linear interactions better

## Feature Importance (Top 10)

From LightGBM + Graph model:

1. `fraud_score` (composite)
2. `shared_devices` (ring detection)
3. `new_device` (ATO signal)
4. `txn_count_1h` (velocity)
5. `amount_zscore` (spike detection)
6. `account_degree` (activity breadth)
7. `merchant_degree` (merchant popularity)
8. `txn_count_24h` (velocity)
9. `new_merchant` (unfamiliar merchant)
10. `high_risk_mcc` (merchant risk)

## Fraud Pattern Detection

Model successfully catches all 4 synthetic patterns:
- **Account Takeover**: Detected via `new_device`, velocity, new merchants
- **Card Testing**: Detected via low-amount velocity + spike
- **Shared-Device Ring**: Detected via `shared_devices`, graph clustering
- **Merchant Concentration**: Detected via `high_risk_mcc`, merchant degree

## Limitations & Caveats

### Data
- **Synthetic patterns**: Real fraud is more nuanced and adversarial
- **Class imbalance**: 1.84% fraud in test; real rates vary (0.1%-5%)
- **Pattern coverage**: Only 4 patterns; real fraud has dozens

### Models
- **Offline evaluation**: No online A/B test or production validation
- **No adversarial adaptation**: Fraudsters adapt; model drift not simulated
- **Graph simplicity**: Real networks have more layers, obfuscation, velocity

### Metrics
- **Precision@K assumes fixed budget**: Real teams adjust thresholds dynamically
- **No cost modeling**: Different fraud types have different losses
- **No review time**: Real analysts spend 2-10 min/case; not factored in

## Deployment Considerations

If this were production:

1. **Alert budget**: Set K based on team capacity (e.g., 100-200 alerts/day)
2. **Operating point**: Choose threshold for 90-95% precision at chosen K
3. **Retraining**: Monthly or when drift detected (PR-AUC drop >5%)
4. **False negative review**: Sample non-flagged transactions for missed fraud
5. **Feedback loop**: Disposition labels retrain model weekly

## Conclusion

Graph features substantially improve fraud detection precision, especially for ring-based and shared-device patterns. LightGBM + Graph achieves near-perfect performance on this synthetic test set (PR-AUC 0.998), demonstrating the value of relational signals.

**Production readiness**: This is a portfolio demo. Real deployment requires:
- Live feature pipelines (streaming graph, velocity)
- Challenger models + A/B testing
- Drift monitoring + retraining
- Adversarial robustness testing

---

**Evaluation Date**: 2026-09-20  
**Data**: Synthetic (80k transactions, 2.5% fraud)  
**Best Model**: LightGBM + Graph (PR-AUC 0.998)
