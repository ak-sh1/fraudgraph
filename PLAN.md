# FraudGraph — Build Package (v1)

**Owner:** Fraud Planner → Fraud Builder  
**Audience:** Akash Gupta — York CS, Toronto; DS/fintech internship portfolio  
**Goal:** Ship a credible transaction fraud / anomaly product demo: graph features + scores + analyst case queue + honest offline metrics. Software + data science hybrid. No fake alpha.

---

## 1. Data (concrete)

### Primary (MVP): synthetic generator
Build a controllable generator so demos and labels are honest and reproducible.

**Entities**
- `accounts` (account_id, open_ts, country, risk_tier, device_ids[])
- `merchants` (merchant_id, mcc, country, high_risk_flag)
- `transactions` (txn_id, ts, amount, account_id, merchant_id, channel, device_id, is_fraud)

**Fraud patterns to inject (labeled)**
1. **Account takeover burst** — sudden new device + high velocity + new merchants  
2. **Card testing** — many small auth-like amounts then a spike  
3. **Shared-device ring** — 3–8 accounts share 1–2 devices; coordinated spend  
4. **Merchant concentration** — mule-like accounts drain to one high-risk MCC cluster  

**Scale (MVP defaults)**
- ~50k–100k txns, ~5k accounts, ~2k merchants, ~90 days  
- Fraud rate ~0.5–2% (realistic imbalance)  
- Seeded RNG; write `data/raw/` + `data/processed/` with a `DATASET.md` describing rates and patterns  

**Optional stretch (not blocking MVP)**
- Public: [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) *or* classic credit-card anomaly CSVs  
- Only if time remains: map public rows into the same feature schema; keep synthetic as the demo path so labels and graph rings stay explainable in interviews  

**Leakage rules (non-negotiable)**
- Time-based split only (train → val → test by `ts`)  
- No future aggregates in features  
- Graph features computed with **train-only edges** for train rows; for val/test use edges available **as of that row's timestamp** (or freeze graph at train end and document the choice)

---

## 2. MVP scope

### A. Graph / relational features
Bipartite / multipartite graph: **accounts ↔ merchants**, plus **accounts ↔ devices**.

Use NetworkX (or igraph) — **not** a GNN-first flex.

**Feature ideas (ship 6–10, document each)**
- Account degree / merchant degree (rolling or as-of)  
- Shared-neighbor count (accounts sharing merchants/devices)  
- Shortest path / common merchants in last N days  
- New-merchant / new-device flags  
- Velocity: txn count & amount sum in 1h / 24h / 7d  
- Amount z-score vs account history  
- MCC rarity / merchant fraud prior (train-only)  

### B. Scoring
1. **Rules baseline** — velocity, new device, high-risk MCC, amount spike (tunable thresholds)  
2. **Tabular model** — LogisticRegression or LightGBM/XGBoost on tabular + graph features  
3. **Optional stretch** — isolation forest / unsupervised score for unlabeled story; keep supervised as primary if labels exist  

Output per txn: `score`, `top_reasons` (SHAP or simple contrib / rule hits), `model_version`.

### C. Analyst case queue UI (demo-critical)
**Stack recommendation (8–10 day ship):**  
- **Python** feature/model pipeline (`src/`)  
- **Next.js** case queue for Vercel deployment
- Optional Streamlit for local dev

**Queue UX**
- List: score, amount, account, merchant, pattern tags, status (`open` / `reviewed` / `confirmed_fraud` / `false_positive`)  
- Detail: txn timeline, graph neighborhood sketch (ego graph), feature values, model reasons  
- Actions: disposition + note; write to `cases.csv` or SQLite  

### D. Baselines to beat / report beside
- Random / volume baseline  
- Rules-only  
- Tabular-without-graph  
- Tabular+graph (**claimed lift only with side-by-side metrics**)

---

## 3. Evals (honest offline)

**Primary**
- **Precision@K** and **Recall@K** (K = expected daily alert budget, e.g. 50 / 100 / 200)  
- **PR-AUC** (better than ROC under imbalance)  
- **ROC-AUC** — report if useful, but do not lead with it alone  
- **Alert fatigue:** alerts/day at fixed recall, FP rate among top-K, analyst "hours" proxy (cases × avg review minutes)

**Protocol**
- Walk-forward or fixed time split (document dates)  
- Calibration / score histogram optional  
- Confusion at chosen operating point for the case queue threshold  

**Artifacts**
- `reports/metrics.json` + short `reports/EVAL.md`  
- One chart: PR curve + Precision@K bar comparing baselines  

---

## 4. What NOT to claim

- Not real bank / card-network fraud detection; **synthetic (or public contest) data only**  
- No guaranteed catch rate, dollar savings, or production FPR  
- No "beats industry SOTA" / unnamed vendor comparisons  
- Graph features are **relational signals**, not proof of organized crime rings in the wild  
- Online / streaming detection, real-time graph DB, and GNN production serving are **out of scope** unless clearly labeled future work  
- Do not invent partner logos, compliance certifications, or "deployed at …"

---

## 5. Suggested repo layout

```
fraudgraph/
  README.md
  PLAN.md          # this file
  DATASET.md
  requirements.txt
  package.json
  src/
    generate/
    features/
    models/
    eval/
  app/             # Next.js case queue
  notebooks/
  data/            # gitignore raw; keep tiny sample
  reports/
  tests/           # leakage + generator invariants
```

## 6. Success criteria (done when)

- [ ] Reproducible synthetic dataset + documented fraud rate  
- [ ] Graph features computed without future leakage  
- [ ] Rules + tabular±graph compared on P@K / PR-AUC  
- [ ] Case queue demo: triage → disposition in <2 minutes  
- [ ] README states limitations and honest metrics  
- [ ] 60–90s demo script Akash can run cold  

**Locked decisions (Fraud Planner, 2026-09-20)**  
- Data: **synthetic-first**  
- UI: **Next.js for Vercel** (Streamlit optional local)  
- Models: **rules + tabular; graph features; no GNN in MVP**  
- Evals: **Precision@K + PR-AUC + time split; ROC-AUC secondary**
