"""
Train fraud detection models:
1. Rules baseline
2. Logistic Regression (with and without graph features)
3. LightGBM (with and without graph features)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, average_precision_score,
    precision_score, recall_score
)
import lightgbm as lgb
import shap

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_features():
    """Load processed features."""
    df = pd.read_csv(PROCESSED_DIR / "transactions_features.csv", parse_dates=["ts"])
    return df


def time_based_split(df, train_frac=0.6, val_frac=0.2):
    """Split by time to avoid leakage."""
    df = df.sort_values("ts").reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    return train, val, test


def rules_baseline(df):
    """Simple rules-based scoring."""
    score = 0.0
    
    # High velocity
    if df["txn_count_1h"] >= 3:
        score += 0.3
    if df["txn_count_24h"] >= 10:
        score += 0.2
    
    # New device
    if df["new_device"] == 1:
        score += 0.2
    
    # High risk MCC
    if df["high_risk_mcc"] == 1:
        score += 0.15
    
    # Amount spike
    if df["amount_zscore"] > 2:
        score += 0.15
    
    # Shared devices (potential ring)
    if df["shared_devices"] > 0:
        score += 0.1
    
    return min(score, 1.0)


def apply_rules_baseline(df):
    """Apply rules to all transactions."""
    scores = []
    for idx, row in df.iterrows():
        scores.append(rules_baseline(row))
    return np.array(scores)


def train_models(train_df, val_df, test_df):
    """Train all models."""
    
    # Define feature sets
    base_features = [
        "amount", "hour", "day_of_week", "is_weekend", "is_online",
        "txn_count_1h", "txn_count_24h", "txn_count_7d",
        "amount_sum_1h", "amount_sum_24h", "amount_sum_7d",
        "amount_zscore", "high_risk_mcc", "mcc"
    ]
    
    graph_features = [
        "account_degree", "merchant_degree", "device_degree",
        "shared_merchants", "shared_devices", "new_merchant", "new_device"
    ]
    
    all_features = base_features + graph_features
    
    # Prepare data
    X_train_base = train_df[base_features].fillna(0)
    X_train_all = train_df[all_features].fillna(0)
    y_train = train_df["is_fraud"].astype(int)
    
    X_val_base = val_df[base_features].fillna(0)
    X_val_all = val_df[all_features].fillna(0)
    y_val = val_df["is_fraud"].astype(int)
    
    X_test_base = test_df[base_features].fillna(0)
    X_test_all = test_df[all_features].fillna(0)
    y_test = test_df["is_fraud"].astype(int)
    
    print("\n=== Rules Baseline ===")
    rules_train_scores = apply_rules_baseline(train_df)
    rules_val_scores = apply_rules_baseline(val_df)
    rules_test_scores = apply_rules_baseline(test_df)
    
    print(f"  Train PR-AUC: {average_precision_score(y_train, rules_train_scores):.4f}")
    print(f"  Val PR-AUC: {average_precision_score(y_val, rules_val_scores):.4f}")
    print(f"  Test PR-AUC: {average_precision_score(y_test, rules_test_scores):.4f}")
    
    # Save rules scores
    with open(MODELS_DIR / "rules_scores.pkl", "wb") as f:
        pickle.dump({
            "train": rules_train_scores,
            "val": rules_val_scores,
            "test": rules_test_scores
        }, f)
    
    print("\n=== Logistic Regression (Base Features) ===")
    lr_base = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_base.fit(X_train_base, y_train)
    
    lr_base_val_scores = lr_base.predict_proba(X_val_base)[:, 1]
    lr_base_test_scores = lr_base.predict_proba(X_test_base)[:, 1]
    
    print(f"  Val PR-AUC: {average_precision_score(y_val, lr_base_val_scores):.4f}")
    print(f"  Test PR-AUC: {average_precision_score(y_test, lr_base_test_scores):.4f}")
    
    with open(MODELS_DIR / "lr_base.pkl", "wb") as f:
        pickle.dump(lr_base, f)
    
    print("\n=== Logistic Regression (Base + Graph Features) ===")
    lr_graph = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_graph.fit(X_train_all, y_train)
    
    lr_graph_val_scores = lr_graph.predict_proba(X_val_all)[:, 1]
    lr_graph_test_scores = lr_graph.predict_proba(X_test_all)[:, 1]
    
    print(f"  Val PR-AUC: {average_precision_score(y_val, lr_graph_val_scores):.4f}")
    print(f"  Test PR-AUC: {average_precision_score(y_test, lr_graph_test_scores):.4f}")
    
    with open(MODELS_DIR / "lr_graph.pkl", "wb") as f:
        pickle.dump(lr_graph, f)
    
    print("\n=== LightGBM (Base Features) ===")
    lgb_base = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        class_weight="balanced",
        random_state=42,
        verbose=-1
    )
    lgb_base.fit(X_train_base, y_train)
    
    lgb_base_val_scores = lgb_base.predict_proba(X_val_base)[:, 1]
    lgb_base_test_scores = lgb_base.predict_proba(X_test_base)[:, 1]
    
    print(f"  Val PR-AUC: {average_precision_score(y_val, lgb_base_val_scores):.4f}")
    print(f"  Test PR-AUC: {average_precision_score(y_test, lgb_base_test_scores):.4f}")
    
    with open(MODELS_DIR / "lgb_base.pkl", "wb") as f:
        pickle.dump(lgb_base, f)
    
    print("\n=== LightGBM (Base + Graph Features) ===")
    lgb_graph = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        class_weight="balanced",
        random_state=42,
        verbose=-1
    )
    lgb_graph.fit(X_train_all, y_train)
    
    lgb_graph_val_scores = lgb_graph.predict_proba(X_val_all)[:, 1]
    lgb_graph_test_scores = lgb_graph.predict_proba(X_test_all)[:, 1]
    
    print(f"  Val PR-AUC: {average_precision_score(y_val, lgb_graph_val_scores):.4f}")
    print(f"  Test PR-AUC: {average_precision_score(y_test, lgb_graph_test_scores):.4f}")
    
    with open(MODELS_DIR / "lgb_graph.pkl", "wb") as f:
        pickle.dump(lgb_graph, f)
    
    # Save feature names
    with open(MODELS_DIR / "feature_names.pkl", "wb") as f:
        pickle.dump({
            "base_features": base_features,
            "graph_features": graph_features,
            "all_features": all_features
        }, f)
    
    print("\nModels saved to:", MODELS_DIR)
    
    return {
        "lr_base": lr_base,
        "lr_graph": lr_graph,
        "lgb_base": lgb_base,
        "lgb_graph": lgb_graph
    }


def main():
    print("Training fraud detection models...")
    
    # Load features
    df = load_features()
    print(f"Loaded {len(df)} transactions with {len(df.columns)} features")
    
    # Time-based split
    print("\nSplitting data by time...")
    train, val, test = time_based_split(df)
    
    print(f"  Train: {len(train)} ({train['is_fraud'].sum()} fraud, {train['is_fraud'].sum()/len(train):.2%})")
    print(f"  Val: {len(val)} ({val['is_fraud'].sum()} fraud, {val['is_fraud'].sum()/len(val):.2%})")
    print(f"  Test: {len(test)} ({test['is_fraud'].sum()} fraud, {test['is_fraud'].sum()/len(test):.2%})")
    
    # Save splits
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    
    # Train models
    models = train_models(train, val, test)
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
