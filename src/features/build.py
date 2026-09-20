"""
Build features for fraud detection with NetworkX graph features.

Features:
1. Basic: amount, hour, day_of_week, mcc
2. Velocity: txn_count_1h, txn_count_24h, amount_sum_7d
3. Graph: account_degree, merchant_degree, shared_neighbors, new_merchant, new_device
4. Risk: high_risk_mcc, amount_zscore
"""

import numpy as np
import pandas as pd
import networkx as nx
from datetime import timedelta
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load raw data."""
    accounts = pd.read_csv(RAW_DIR / "accounts.csv", parse_dates=["open_ts"])
    merchants = pd.read_csv(RAW_DIR / "merchants.csv")
    txns = pd.read_csv(RAW_DIR / "transactions.csv")
    txns["ts"] = pd.to_datetime(txns["ts"], format="mixed")
    return accounts, merchants, txns


def add_basic_features(txns_df):
    """Add basic transaction features."""
    txns_df["hour"] = txns_df["ts"].dt.hour
    txns_df["day_of_week"] = txns_df["ts"].dt.dayofweek
    txns_df["is_weekend"] = txns_df["day_of_week"].isin([5, 6]).astype(int)
    txns_df["is_online"] = (txns_df["channel"] == "online").astype(int)
    return txns_df


def add_velocity_features(txns_df):
    """Add velocity features (time-aware, no future leakage)."""
    txns_df = txns_df.sort_values("ts").reset_index(drop=True)
    
    # Initialize velocity features
    txns_df["txn_count_1h"] = 0
    txns_df["txn_count_24h"] = 0
    txns_df["txn_count_7d"] = 0
    txns_df["amount_sum_1h"] = 0.0
    txns_df["amount_sum_24h"] = 0.0
    txns_df["amount_sum_7d"] = 0.0
    
    # Account history tracking
    account_history = defaultdict(list)
    
    for idx, row in txns_df.iterrows():
        account_id = row["account_id"]
        ts = row["ts"]
        amount = row["amount"]
        
        # Get prior transactions for this account
        history = account_history[account_id]
        
        # Filter by time windows
        recent_1h = [(t, a) for t, a in history if (ts - t).total_seconds() <= 3600]
        recent_24h = [(t, a) for t, a in history if (ts - t).total_seconds() <= 86400]
        recent_7d = [(t, a) for t, a in history if (ts - t).total_seconds() <= 604800]
        
        # Update features
        txns_df.at[idx, "txn_count_1h"] = len(recent_1h)
        txns_df.at[idx, "txn_count_24h"] = len(recent_24h)
        txns_df.at[idx, "txn_count_7d"] = len(recent_7d)
        txns_df.at[idx, "amount_sum_1h"] = sum(a for _, a in recent_1h)
        txns_df.at[idx, "amount_sum_24h"] = sum(a for _, a in recent_24h)
        txns_df.at[idx, "amount_sum_7d"] = sum(a for _, a in recent_7d)
        
        # Add current transaction to history
        account_history[account_id].append((ts, amount))
    
    return txns_df


def add_amount_features(txns_df):
    """Add amount-based features."""
    txns_df = txns_df.sort_values("ts").reset_index(drop=True)
    
    txns_df["amount_zscore"] = 0.0
    
    account_amounts = defaultdict(list)
    
    for idx, row in txns_df.iterrows():
        account_id = row["account_id"]
        amount = row["amount"]
        
        history = account_amounts[account_id]
        
        if len(history) >= 3:
            mean_amt = np.mean(history)
            std_amt = np.std(history)
            if std_amt > 0:
                zscore = (amount - mean_amt) / std_amt
                txns_df.at[idx, "amount_zscore"] = zscore
        
        account_amounts[account_id].append(amount)
    
    return txns_df


def build_graph_features(txns_df, merchants_df, train_cutoff_idx=None):
    """
    Build graph features using NetworkX.
    
    If train_cutoff_idx is provided, only use transactions up to that index
    for graph construction (to avoid leakage).
    """
    txns_df = txns_df.sort_values("ts").reset_index(drop=True)
    
    # Initialize graph features
    txns_df["account_degree"] = 0
    txns_df["merchant_degree"] = 0
    txns_df["device_degree"] = 0
    txns_df["shared_merchants"] = 0
    txns_df["shared_devices"] = 0
    txns_df["new_merchant"] = 1
    txns_df["new_device"] = 1
    
    # Track edges over time
    account_merchants = defaultdict(set)
    account_devices = defaultdict(set)
    merchant_accounts = defaultdict(set)
    device_accounts = defaultdict(set)
    
    # If train_cutoff provided, pre-build graph from training data
    if train_cutoff_idx is not None:
        train_txns = txns_df.iloc[:train_cutoff_idx]
        for _, row in train_txns.iterrows():
            account_merchants[row["account_id"]].add(row["merchant_id"])
            account_devices[row["account_id"]].add(row["device_id"])
            merchant_accounts[row["merchant_id"]].add(row["account_id"])
            device_accounts[row["device_id"]].add(row["account_id"])
    
    # Process each transaction
    for idx, row in txns_df.iterrows():
        account_id = row["account_id"]
        merchant_id = row["merchant_id"]
        device_id = row["device_id"]
        
        # Compute features based on graph state BEFORE this transaction
        
        # Degree features
        txns_df.at[idx, "account_degree"] = len(account_merchants[account_id])
        txns_df.at[idx, "merchant_degree"] = len(merchant_accounts[merchant_id])
        txns_df.at[idx, "device_degree"] = len(device_accounts[device_id])
        
        # Shared neighbor features
        # Accounts sharing this merchant
        merchant_neighbors = merchant_accounts[merchant_id]
        if len(merchant_neighbors) > 0:
            shared_count = sum(
                len(account_merchants[acc].intersection(account_merchants[account_id]))
                for acc in merchant_neighbors
                if acc != account_id
            )
            txns_df.at[idx, "shared_merchants"] = shared_count
        
        # Accounts sharing this device
        device_neighbors = device_accounts[device_id]
        if len(device_neighbors) > 0:
            shared_count = sum(
                len(account_devices[acc].intersection(account_devices[account_id]))
                for acc in device_neighbors
                if acc != account_id
            )
            txns_df.at[idx, "shared_devices"] = shared_count
        
        # New merchant/device flags
        txns_df.at[idx, "new_merchant"] = int(merchant_id not in account_merchants[account_id])
        txns_df.at[idx, "new_device"] = int(device_id not in account_devices[account_id])
        
        # Update graph (only if not frozen at train cutoff)
        if train_cutoff_idx is None or idx < train_cutoff_idx:
            account_merchants[account_id].add(merchant_id)
            account_devices[account_id].add(device_id)
            merchant_accounts[merchant_id].add(account_id)
            device_accounts[device_id].add(account_id)
    
    return txns_df


def add_merchant_features(txns_df, merchants_df):
    """Add merchant-based features."""
    txns_df = txns_df.merge(
        merchants_df[["merchant_id", "mcc", "high_risk_flag"]],
        on="merchant_id",
        how="left"
    )
    txns_df["high_risk_mcc"] = txns_df["high_risk_flag"].astype(int)
    return txns_df


def main():
    print("Building features...")
    
    # Load data
    accounts, merchants, txns = load_data()
    
    print(f"  Loaded {len(txns)} transactions")
    
    # Add features
    print("  - Basic features...")
    txns = add_basic_features(txns)
    
    print("  - Merchant features...")
    txns = add_merchant_features(txns, merchants)
    
    print("  - Velocity features...")
    txns = add_velocity_features(txns)
    
    print("  - Amount features...")
    txns = add_amount_features(txns)
    
    print("  - Graph features...")
    txns = build_graph_features(txns, merchants)
    
    # Save
    output_path = PROCESSED_DIR / "transactions_features.csv"
    txns.to_csv(output_path, index=False)
    
    print(f"\nFeatures built: {len(txns.columns)} columns")
    print(f"Output: {output_path}")
    
    # Show sample
    feature_cols = [
        "amount", "hour", "txn_count_1h", "txn_count_24h", 
        "amount_zscore", "account_degree", "merchant_degree",
        "shared_merchants", "shared_devices", "new_merchant", "new_device",
        "high_risk_mcc", "is_fraud"
    ]
    print("\nSample features:")
    print(txns[feature_cols].head(10))


if __name__ == "__main__":
    main()
