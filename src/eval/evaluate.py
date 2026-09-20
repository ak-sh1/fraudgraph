"""
Evaluate fraud detection models with Precision@K, Recall@K, PR-AUC.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = Path(__file__).parent.parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_test_data():
    """Load test set."""
    test = pd.read_csv(PROCESSED_DIR / "test.csv", parse_dates=["ts"])
    return test


def load_models_and_scores():
    """Load trained models and scores."""
    with open(MODELS_DIR / "lr_base.pkl", "rb") as f:
        lr_base = pickle.load(f)
    
    with open(MODELS_DIR / "lr_graph.pkl", "rb") as f:
        lr_graph = pickle.load(f)
    
    with open(MODELS_DIR / "lgb_base.pkl", "rb") as f:
        lgb_base = pickle.load(f)
    
    with open(MODELS_DIR / "lgb_graph.pkl", "rb") as f:
        lgb_graph = pickle.load(f)
    
    with open(MODELS_DIR / "rules_scores.pkl", "rb") as f:
        rules_scores = pickle.load(f)
    
    with open(MODELS_DIR / "feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    
    return {
        "models": {
            "lr_base": lr_base,
            "lr_graph": lr_graph,
            "lgb_base": lgb_base,
            "lgb_graph": lgb_graph
        },
        "rules_scores": rules_scores,
        "feature_names": feature_names
    }


def precision_at_k(y_true, scores, k):
    """Compute Precision@K."""
    top_k_idx = np.argsort(scores)[::-1][:k]
    return precision_score(y_true[top_k_idx], [1] * k, zero_division=0)


def recall_at_k(y_true, scores, k):
    """Compute Recall@K."""
    top_k_idx = np.argsort(scores)[::-1][:k]
    return recall_score(y_true[top_k_idx], [1] * k, zero_division=0)


def evaluate_model(y_true, scores, model_name, k_values=[50, 100, 200]):
    """Evaluate a single model."""
    metrics = {
        "model": model_name,
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }
    
    for k in k_values:
        k = min(k, len(y_true))
        metrics[f"precision@{k}"] = float(precision_at_k(y_true, scores, k))
        metrics[f"recall@{k}"] = float(recall_at_k(y_true, scores, k))
    
    return metrics


def plot_pr_curves(results, y_true_test):
    """Plot PR curves for all models."""
    plt.figure(figsize=(10, 6))
    
    colors = {
        "rules": "gray",
        "lr_base": "blue",
        "lr_graph": "darkblue",
        "lgb_base": "orange",
        "lgb_graph": "red"
    }
    
    for result in results:
        model_name = result["model"]
        scores = result["scores"]
        
        precision, recall, _ = precision_recall_curve(y_true_test, scores)
        
        label = f"{model_name} (PR-AUC={result['pr_auc']:.3f})"
        plt.plot(recall, precision, label=label, color=colors.get(model_name, "black"))
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(REPORTS_DIR / "pr_curves.png", dpi=150)
    print(f"  Saved: {REPORTS_DIR / 'pr_curves.png'}")


def plot_precision_at_k(results, k_values=[50, 100, 200]):
    """Plot Precision@K comparison."""
    plt.figure(figsize=(10, 6))
    
    models = [r["model"] for r in results]
    x = np.arange(len(k_values))
    width = 0.15
    
    colors = ["gray", "blue", "darkblue", "orange", "red"]
    
    for i, result in enumerate(results):
        precisions = [result.get(f"precision@{k}", 0) for k in k_values]
        plt.bar(x + i * width, precisions, width, label=result["model"], color=colors[i])
    
    plt.xlabel("K (Alert Budget)")
    plt.ylabel("Precision")
    plt.title("Precision@K Comparison")
    plt.xticks(x + width * 2, [f"{k}" for k in k_values])
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    
    plt.savefig(REPORTS_DIR / "precision_at_k.png", dpi=150)
    print(f"  Saved: {REPORTS_DIR / 'precision_at_k.png'}")


def main():
    print("Evaluating models...")
    
    # Load test data
    test = load_test_data()
    y_true = test["is_fraud"].values
    
    print(f"  Test set: {len(test)} transactions, {y_true.sum()} fraud ({y_true.sum()/len(test):.2%})")
    
    # Load models
    artifacts = load_models_and_scores()
    models = artifacts["models"]
    rules_scores = artifacts["rules_scores"]
    feature_names = artifacts["feature_names"]
    
    # Get test scores
    X_test_base = test[feature_names["base_features"]].fillna(0)
    X_test_all = test[feature_names["all_features"]].fillna(0)
    
    results = []
    
    # Rules
    print("\n  Evaluating rules baseline...")
    rules_test_scores = rules_scores["test"]
    results.append({
        **evaluate_model(y_true, rules_test_scores, "rules"),
        "scores": rules_test_scores
    })
    
    # LR Base
    print("  Evaluating LR (base)...")
    lr_base_scores = models["lr_base"].predict_proba(X_test_base)[:, 1]
    results.append({
        **evaluate_model(y_true, lr_base_scores, "lr_base"),
        "scores": lr_base_scores
    })
    
    # LR Graph
    print("  Evaluating LR (graph)...")
    lr_graph_scores = models["lr_graph"].predict_proba(X_test_all)[:, 1]
    results.append({
        **evaluate_model(y_true, lr_graph_scores, "lr_graph"),
        "scores": lr_graph_scores
    })
    
    # LGB Base
    print("  Evaluating LightGBM (base)...")
    lgb_base_scores = models["lgb_base"].predict_proba(X_test_base)[:, 1]
    results.append({
        **evaluate_model(y_true, lgb_base_scores, "lgb_base"),
        "scores": lgb_base_scores
    })
    
    # LGB Graph
    print("  Evaluating LightGBM (graph)...")
    lgb_graph_scores = models["lgb_graph"].predict_proba(X_test_all)[:, 1]
    results.append({
        **evaluate_model(y_true, lgb_graph_scores, "lgb_graph"),
        "scores": lgb_graph_scores
    })
    
    # Save metrics
    metrics_output = {
        "test_size": len(test),
        "fraud_count": int(y_true.sum()),
        "fraud_rate": float(y_true.sum() / len(test)),
        "models": [{k: v for k, v in r.items() if k != "scores"} for r in results]
    }
    
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=2)
    
    print(f"\n  Saved: {REPORTS_DIR / 'metrics.json'}")
    
    # Print comparison table
    print("\n=== Test Set Results ===\n")
    print(f"{'Model':<15} {'PR-AUC':<10} {'ROC-AUC':<10} {'P@50':<10} {'P@100':<10} {'P@200':<10}")
    print("-" * 75)
    for r in results:
        print(f"{r['model']:<15} {r['pr_auc']:<10.4f} {r['roc_auc']:<10.4f} "
              f"{r.get('precision@50', 0):<10.4f} {r.get('precision@100', 0):<10.4f} "
              f"{r.get('precision@200', 0):<10.4f}")
    
    # Plots
    print("\n  Generating plots...")
    plot_pr_curves(results, y_true)
    plot_precision_at_k(results)
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
