"""
Export demo artifacts for Next.js UI.

Generates:
- alerts.json: Top-K alerts with scores and reasons
- graph_data.json: Ego-graph data for visualization
"""

import json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
DEMO_DIR = DATA_DIR / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)


def load_model_and_scores():
    """Load best model (LightGBM + graph)."""
    with open(MODELS_DIR / "lgb_graph.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open(MODELS_DIR / "feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    
    return model, feature_names


def compute_reasons(row, feature_names):
    """Compute top reasons for a transaction."""
    reasons = []
    
    # High-scoring conditions
    if row["txn_count_1h"] >= 3:
        reasons.append({"feature": "High 1h velocity", "value": f"{row['txn_count_1h']} txns"})
    
    if row["txn_count_24h"] >= 10:
        reasons.append({"feature": "High 24h velocity", "value": f"{row['txn_count_24h']} txns"})
    
    if row["new_device"] == 1:
        reasons.append({"feature": "New device", "value": "First time"})
    
    if row["new_merchant"] == 1:
        reasons.append({"feature": "New merchant", "value": "First time"})
    
    if row["shared_devices"] > 0:
        reasons.append({"feature": "Shared device ring", "value": f"{int(row['shared_devices'])} shared"})
    
    if row["high_risk_mcc"] == 1:
        reasons.append({"feature": "High-risk MCC", "value": f"MCC {int(row['mcc'])}"})
    
    if row["amount_zscore"] > 2:
        reasons.append({"feature": "Amount spike", "value": f"z-score {row['amount_zscore']:.1f}"})
    
    if row["account_degree"] > 20:
        reasons.append({"feature": "High account activity", "value": f"{int(row['account_degree'])} merchants"})
    
    return reasons[:5]  # Top 5


def build_ego_graph(txns_df, center_txn):
    """Build ego-graph around a transaction."""
    account_id = center_txn["account_id"]
    merchant_id = center_txn["merchant_id"]
    device_id = center_txn["device_id"]
    
    # Find related transactions (same account, merchant, or device)
    related_mask = (
        (txns_df["account_id"] == account_id) |
        (txns_df["merchant_id"] == merchant_id) |
        (txns_df["device_id"] == device_id)
    )
    
    related_txns = txns_df[related_mask].copy()
    
    # Nodes
    nodes = []
    edges = []
    
    # Center account
    nodes.append({
        "id": account_id,
        "type": "account",
        "label": account_id,
        "is_center": True
    })
    
    # Related accounts
    related_accounts = related_txns["account_id"].unique()
    for acc in related_accounts:
        if acc != account_id:
            nodes.append({
                "id": acc,
                "type": "account",
                "label": acc,
                "is_center": False
            })
    
    # Merchants
    related_merchants = related_txns["merchant_id"].unique()
    for mer in related_merchants:
        nodes.append({
            "id": mer,
            "type": "merchant",
            "label": mer,
            "is_center": mer == merchant_id
        })
    
    # Devices
    related_devices = related_txns["device_id"].unique()
    for dev in related_devices:
        nodes.append({
            "id": dev,
            "type": "device",
            "label": dev,
            "is_center": dev == device_id
        })
    
    # Edges
    for _, txn in related_txns.iterrows():
        edges.append({
            "source": txn["account_id"],
            "target": txn["merchant_id"],
            "type": "transaction"
        })
        edges.append({
            "source": txn["account_id"],
            "target": txn["device_id"],
            "type": "device_usage"
        })
    
    return {
        "nodes": nodes,
        "edges": edges
    }


def main():
    print("Exporting demo artifacts...")
    
    # Load test data
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    test["ts"] = pd.to_datetime(test["ts"], format="mixed")
    print(f"  Loaded {len(test)} test transactions")
    
    # Load model
    model, feature_names = load_model_and_scores()
    
    # Score test transactions
    X_test = test[feature_names["all_features"]].fillna(0)
    test["fraud_score"] = model.predict_proba(X_test)[:, 1]
    
    # Get top-K alerts
    K = 200
    top_alerts = test.nlargest(K, "fraud_score").copy()
    
    print(f"  Selected top {K} alerts")
    
    # Build alert records
    alerts = []
    for idx, row in top_alerts.iterrows():
        reasons = compute_reasons(row, feature_names)
        
        alert = {
            "txn_id": row["txn_id"],
            "timestamp": row["ts"].isoformat(),
            "amount": float(row["amount"]),
            "account_id": row["account_id"],
            "merchant_id": row["merchant_id"],
            "device_id": row["device_id"],
            "channel": row["channel"],
            "fraud_score": float(row["fraud_score"]),
            "is_fraud": bool(row["is_fraud"]),
            "fraud_pattern": row["fraud_pattern"] if pd.notna(row["fraud_pattern"]) else None,
            "reasons": reasons,
            "status": "open"
        }
        alerts.append(alert)
    
    # Save alerts
    with open(DEMO_DIR / "alerts.json", "w") as f:
        json.dump(alerts, f, indent=2)
    
    print(f"  Saved: {DEMO_DIR / 'alerts.json'}")
    
    # Build ego-graphs for top 20
    print("  Building ego-graphs for top 20...")
    ego_graphs = {}
    
    for alert in alerts[:20]:
        txn_id = alert["txn_id"]
        center_txn = test[test["txn_id"] == txn_id].iloc[0]
        ego_graph = build_ego_graph(test, center_txn)
        ego_graphs[txn_id] = ego_graph
    
    with open(DEMO_DIR / "ego_graphs.json", "w") as f:
        json.dump(ego_graphs, f, indent=2)
    
    print(f"  Saved: {DEMO_DIR / 'ego_graphs.json'}")
    
    print("\n✓ Demo export complete!")
    print(f"  {len(alerts)} alerts exported")
    print(f"  {len(ego_graphs)} ego-graphs exported")


if __name__ == "__main__":
    main()
