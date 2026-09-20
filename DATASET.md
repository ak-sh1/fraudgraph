# FraudGraph Dataset

## Overview

Synthetic transaction dataset generated with controlled fraud patterns for fraud detection research and demonstration.

## Statistics

- **Total Transactions**: ~80,000
- **Accounts**: 5,000
- **Merchants**: 2,000
- **Time Period**: 90 days (June 1 - Aug 28, 2026)
- **Fraud Rate**: ~2.5% (realistic imbalance)

## Fraud Patterns (Labeled)

### 1. Account Takeover (ATO) Burst
- **Count**: ~500 transactions
- **Pattern**: Sudden new device + high velocity (5-15 txns in 1-2 hours) + new merchants
- **Detection signals**: New device flag, velocity spike, unfamiliar merchant set

### 2. Card Testing
- **Count**: ~560 transactions  
- **Pattern**: Many small auth-like amounts ($0.50-$5) followed by a larger spike ($500-$3000)
- **Detection signals**: Low-amount velocity, then amount spike, same device

### 3. Shared-Device Ring
- **Count**: ~620 transactions
- **Pattern**: 3-8 accounts share 1-2 devices with coordinated spending
- **Detection signals**: High shared-device count, graph clustering, synchronized activity

### 4. Merchant Concentration
- **Count**: ~340 transactions
- **Pattern**: Mule-like accounts drain to one high-risk MCC merchant cluster
- **Detection signals**: High-risk MCC, merchant concentration, repeated drain pattern

## Normal Transactions

- **Count**: ~78,800
- **Distribution**: Log-normal amount distribution (mean ~$33, realistic variance)
- **Channels**: 60% online, 30% POS, 10% mobile
- **Behavior**: Natural shopping patterns, 1-2 devices per account, diverse merchants

## Features

### Basic (Tabular)
- amount, hour, day_of_week, is_weekend, is_online
- mcc, high_risk_mcc

### Velocity (Time-aware)
- txn_count_1h, txn_count_24h, txn_count_7d
- amount_sum_1h, amount_sum_24h, amount_sum_7d
- amount_zscore (vs account history)

### Graph (NetworkX)
- account_degree, merchant_degree, device_degree
- shared_merchants, shared_devices
- new_merchant, new_device (first-time flags)

## Data Splits

- **Train**: 60% (oldest)
- **Validation**: 20%
- **Test**: 20% (most recent)

Time-based split to avoid leakage. Graph features computed using only past edges at each transaction timestamp.

## Files

- `data/raw/accounts.csv` - Account profiles
- `data/raw/merchants.csv` - Merchant profiles  
- `data/raw/transactions.csv` - Full transaction log
- `data/processed/transactions_features.csv` - Transactions with all features
- `data/processed/{train,val,test}.csv` - Time-based splits
- `data/demo/alerts.json` - Top-K alerts for Next.js UI

## Generation

Dataset is fully reproducible with seed=42. To regenerate:

```bash
python -m src.generate.synthetic
python -m src.features.build
```

## Limitations

- **Synthetic data**: Patterns are simplified; real fraud is more nuanced
- **No real PII**: All IDs, amounts, and patterns are synthetic
- **Graph simplicity**: Real fraud networks have more layers and obfuscation
- **Static snapshot**: Real systems need streaming/online detection

## Citation

This dataset is for portfolio demonstration only. Do not use for production fraud detection or publish as a "real" fraud dataset.
