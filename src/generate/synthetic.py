"""
Synthetic transaction generator with labeled fraud patterns.

Generates:
- accounts, merchants, transactions
- 4 fraud patterns: ATO burst, card testing, shared-device ring, merchant concentration
- Realistic 0.5-2% fraud rate, ~80k transactions
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Seeded RNG
SEED = 42
rng = np.random.default_rng(SEED)

# Scale parameters
N_ACCOUNTS = 5000
N_MERCHANTS = 2000
N_TRANSACTIONS = 80000
FRAUD_RATE = 0.015  # 1.5%
DAYS = 90

# Output paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def generate_accounts(n=N_ACCOUNTS):
    """Generate account profiles."""
    countries = ["US", "CA", "GB", "DE", "FR", "AU"]
    risk_tiers = ["low", "medium", "high"]
    
    start_date = datetime(2026, 6, 1)
    
    accounts = []
    for i in range(n):
        open_ts = start_date + timedelta(days=int(rng.integers(-365, 0)))
        accounts.append({
            "account_id": f"ACC{i:05d}",
            "open_ts": open_ts,
            "country": rng.choice(countries),
            "risk_tier": rng.choice(risk_tiers, p=[0.7, 0.25, 0.05]),
        })
    
    return pd.DataFrame(accounts)


def generate_merchants(n=N_MERCHANTS):
    """Generate merchant profiles."""
    # MCC codes (simplified)
    mccs = {
        "retail": [5411, 5912, 5999],  # Grocery, pharmacy, misc
        "restaurant": [5812, 5814],
        "travel": [3000, 4511],
        "online": [5816, 5817],
        "high_risk": [7995, 5967],  # Gambling, e-commerce
    }
    
    mcc_pool = []
    high_risk_flags = []
    for category, codes in mccs.items():
        for _ in range(n // 5):
            mcc_pool.append(rng.choice(codes))
            high_risk_flags.append(category == "high_risk")
    
    # Pad if needed
    while len(mcc_pool) < n:
        mcc_pool.append(5999)
        high_risk_flags.append(False)
    
    mcc_pool = mcc_pool[:n]
    high_risk_flags = high_risk_flags[:n]
    
    merchants = []
    for i in range(n):
        merchants.append({
            "merchant_id": f"MER{i:04d}",
            "mcc": mcc_pool[i],
            "country": rng.choice(["US", "CA", "GB", "DE", "CN", "IN"]),
            "high_risk_flag": high_risk_flags[i],
        })
    
    return pd.DataFrame(merchants)


def generate_normal_transactions(accounts_df, merchants_df, n_normal):
    """Generate normal (non-fraud) transactions."""
    start_ts = datetime(2026, 6, 1)
    end_ts = start_ts + timedelta(days=DAYS)
    
    transactions = []
    device_pool = [f"DEV{i:05d}" for i in range(N_ACCOUNTS * 2)]
    
    for i in range(n_normal):
        account = accounts_df.sample(1, random_state=rng.integers(0, 1e9)).iloc[0]
        merchant = merchants_df.sample(1, random_state=rng.integers(0, 1e9)).iloc[0]
        
        # Random timestamp
        ts = start_ts + timedelta(
            seconds=int(rng.integers(0, int((end_ts - start_ts).total_seconds())))
        )
        
        # Amount distribution (log-normal)
        amount = np.exp(rng.normal(3.5, 1.2))  # Mean ~$33, std varies
        amount = max(1.0, min(5000.0, amount))
        
        # Device: accounts usually have 1-2 devices
        device_id = rng.choice(device_pool[:N_ACCOUNTS])
        
        transactions.append({
            "txn_id": f"TXN{i:06d}",
            "ts": ts,
            "amount": round(amount, 2),
            "account_id": account["account_id"],
            "merchant_id": merchant["merchant_id"],
            "channel": rng.choice(["online", "pos", "mobile"], p=[0.6, 0.3, 0.1]),
            "device_id": device_id,
            "is_fraud": False,
            "fraud_pattern": None,
        })
    
    return transactions


def inject_ato_burst(accounts_df, merchants_df, n_patterns=50):
    """
    Pattern 1: Account takeover burst.
    - Sudden new device
    - High velocity (5-15 txns in 1-2 hours)
    - New merchants
    """
    transactions = []
    start_ts = datetime(2026, 6, 15)
    end_ts = datetime(2026, 8, 15)
    
    for i in range(n_patterns):
        account = accounts_df.sample(1, random_state=int(rng.integers(0, 1e9))).iloc[0]
        new_device = f"DEV_ATO_{i:04d}"
        
        burst_start = start_ts + timedelta(
            seconds=int(rng.integers(0, int((end_ts - start_ts).total_seconds())))
        )
        
        n_txns = int(rng.integers(5, 15))
        for j in range(n_txns):
            merchant = merchants_df.sample(1, random_state=int(rng.integers(0, 1e9))).iloc[0]
            ts = burst_start + timedelta(minutes=int(rng.integers(0, 120)))
            amount = rng.uniform(50, 1500)
            
            transactions.append({
                "txn_id": f"TXN_ATO_{i:04d}_{j:02d}",
                "ts": ts,
                "amount": round(amount, 2),
                "account_id": account["account_id"],
                "merchant_id": merchant["merchant_id"],
                "channel": "online",
                "device_id": new_device,
                "is_fraud": True,
                "fraud_pattern": "ato_burst",
            })
    
    return transactions


def inject_card_testing(accounts_df, merchants_df, n_patterns=40):
    """
    Pattern 2: Card testing.
    - Many small transactions (~$1-5)
    - Then a larger spike
    """
    transactions = []
    start_ts = datetime(2026, 6, 10)
    end_ts = datetime(2026, 8, 20)
    
    for i in range(n_patterns):
        account = accounts_df.sample(1, random_state=rng.integers(0, 1e9)).iloc[0]
        device = f"DEV_TEST_{i:04d}"
        
        test_start = start_ts + timedelta(
            seconds=int(rng.integers(0, int((end_ts - start_ts).total_seconds())))
        )
        
        # Small test transactions
        n_tests = int(rng.integers(8, 20))
        for j in range(n_tests):
            merchant = merchants_df.sample(1, random_state=rng.integers(0, 1e9)).iloc[0]
            ts = test_start + timedelta(minutes=int(rng.integers(0, 60)))
            amount = rng.uniform(0.5, 5.0)
            
            transactions.append({
                "txn_id": f"TXN_TEST_{i:04d}_{j:02d}",
                "ts": ts,
                "amount": round(amount, 2),
                "account_id": account["account_id"],
                "merchant_id": merchant["merchant_id"],
                "channel": "online",
                "device_id": device,
                "is_fraud": True,
                "fraud_pattern": "card_testing",
            })
        
        # Spike transaction
        merchant = merchants_df.sample(1, random_state=rng.integers(0, 1e9)).iloc[0]
        spike_ts = test_start + timedelta(hours=rng.uniform(1, 4))
        spike_amount = rng.uniform(500, 3000)
        
        transactions.append({
            "txn_id": f"TXN_TEST_{i:04d}_SPIKE",
            "ts": spike_ts,
            "amount": round(spike_amount, 2),
            "account_id": account["account_id"],
            "merchant_id": merchant["merchant_id"],
            "channel": "online",
            "device_id": device,
            "is_fraud": True,
            "fraud_pattern": "card_testing",
        })
    
    return transactions


def inject_shared_device_ring(accounts_df, merchants_df, n_rings=15):
    """
    Pattern 3: Shared-device ring.
    - 3-8 accounts share 1-2 devices
    - Coordinated spending
    """
    transactions = []
    start_ts = datetime(2026, 6, 20)
    end_ts = datetime(2026, 8, 25)
    
    for i in range(n_rings):
        ring_size = int(rng.integers(3, 8))
        ring_accounts = accounts_df.sample(ring_size, random_state=rng.integers(0, 1e9))
        shared_devices = [f"DEV_RING_{i:04d}_A", f"DEV_RING_{i:04d}_B"]
        
        # Each account makes 5-10 transactions from shared devices
        for idx, account in ring_accounts.iterrows():
            n_txns = int(rng.integers(5, 10))
            for j in range(n_txns):
                merchant = merchants_df.sample(1, random_state=rng.integers(0, 1e9)).iloc[0]
                ts = start_ts + timedelta(
                    seconds=int(rng.integers(0, int((end_ts - start_ts).total_seconds())))
                )
                amount = rng.uniform(20, 800)
                device = rng.choice(shared_devices)
                
                transactions.append({
                    "txn_id": f"TXN_RING_{i:04d}_{account['account_id']}_{j:02d}",
                    "ts": ts,
                    "amount": round(amount, 2),
                    "account_id": account["account_id"],
                    "merchant_id": merchant["merchant_id"],
                    "channel": rng.choice(["online", "mobile"]),
                    "device_id": device,
                    "is_fraud": True,
                    "fraud_pattern": "shared_device_ring",
                })
    
    return transactions


def inject_merchant_concentration(accounts_df, merchants_df, n_patterns=30):
    """
    Pattern 4: Merchant concentration.
    - Mule-like accounts drain to 1-2 high-risk merchants
    """
    transactions = []
    start_ts = datetime(2026, 6, 25)
    end_ts = datetime(2026, 8, 28)
    
    high_risk_merchants = merchants_df[merchants_df["high_risk_flag"]]
    
    for i in range(n_patterns):
        account = accounts_df.sample(1, random_state=rng.integers(0, 1e9)).iloc[0]
        target_merchant = high_risk_merchants.sample(1, random_state=rng.integers(0, 1e9)).iloc[0]
        device = f"DEV_MULE_{i:04d}"
        
        # 8-15 transactions to same/similar merchants
        n_txns = int(rng.integers(8, 15))
        for j in range(n_txns):
            ts = start_ts + timedelta(
                seconds=int(rng.integers(0, int((end_ts - start_ts).total_seconds())))
            )
            amount = rng.uniform(100, 2000)
            
            transactions.append({
                "txn_id": f"TXN_CONC_{i:04d}_{j:02d}",
                "ts": ts,
                "amount": round(amount, 2),
                "account_id": account["account_id"],
                "merchant_id": target_merchant["merchant_id"],
                "channel": "online",
                "device_id": device,
                "is_fraud": True,
                "fraud_pattern": "merchant_concentration",
            })
    
    return transactions


def main():
    print("Generating synthetic dataset...")
    
    # Generate entities
    print("  - Accounts...")
    accounts_df = generate_accounts()
    accounts_df.to_csv(RAW_DIR / "accounts.csv", index=False)
    
    print("  - Merchants...")
    merchants_df = generate_merchants()
    merchants_df.to_csv(RAW_DIR / "merchants.csv", index=False)
    
    # Calculate fraud/normal split
    n_fraud = int(N_TRANSACTIONS * FRAUD_RATE)
    n_normal = N_TRANSACTIONS - n_fraud
    
    print(f"  - Transactions ({n_normal} normal, {n_fraud} fraud)...")
    
    # Normal transactions
    normal_txns = generate_normal_transactions(accounts_df, merchants_df, n_normal)
    
    # Fraud patterns
    ato_txns = inject_ato_burst(accounts_df, merchants_df, n_patterns=50)
    test_txns = inject_card_testing(accounts_df, merchants_df, n_patterns=40)
    ring_txns = inject_shared_device_ring(accounts_df, merchants_df, n_rings=15)
    conc_txns = inject_merchant_concentration(accounts_df, merchants_df, n_patterns=30)
    
    # Combine
    all_txns = normal_txns + ato_txns + test_txns + ring_txns + conc_txns
    txns_df = pd.DataFrame(all_txns)
    
    # Sort by timestamp
    txns_df = txns_df.sort_values("ts").reset_index(drop=True)
    
    # Re-assign txn_id in order
    txns_df["txn_id"] = [f"TXN{i:06d}" for i in range(len(txns_df))]
    
    txns_df.to_csv(RAW_DIR / "transactions.csv", index=False)
    
    # Stats
    fraud_count = txns_df["is_fraud"].sum()
    fraud_rate = fraud_count / len(txns_df)
    
    print(f"\nDataset generated:")
    print(f"  Accounts: {len(accounts_df)}")
    print(f"  Merchants: {len(merchants_df)}")
    print(f"  Transactions: {len(txns_df)}")
    print(f"  Fraud: {fraud_count} ({fraud_rate:.2%})")
    print(f"  Patterns: ATO={len(ato_txns)}, Testing={len(test_txns)}, Ring={len(ring_txns)}, Concentration={len(conc_txns)}")
    print(f"\nFiles written to: {RAW_DIR}")


if __name__ == "__main__":
    main()
