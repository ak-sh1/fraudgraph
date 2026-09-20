# FraudGraph

**Transaction fraud detection with graph features, rule+ML baselines, and analyst case queue.**

## Overview

FraudGraph demonstrates transaction anomaly detection using:
- **Synthetic fraud patterns**: Account takeover, card testing, shared-device rings, merchant concentration
- **Graph features**: NetworkX bipartite graph (accounts ↔ merchants ↔ devices) 
- **Models**: Rules baseline + tabular (LogReg/LightGBM) with and without graph features
- **Case queue UI**: Next.js app for alert triage with ego-graph visualization
- **Honest metrics**: Precision@K, Recall@K, PR-AUC with time-based splits

## Quick Start

### 1. Generate Data & Train Models (Python)

```bash
# Install Python dependencies
pip install -r requirements.txt

# Generate synthetic dataset (~80k transactions)
python -m src.generate.synthetic

# Build features
python -m src.features.build

# Train models
python -m src.models.train

# Run evaluation
python -m src.eval.evaluate

# Export demo artifacts for Next.js
python -m src.export_demo
```

### 2. Run Case Queue UI (Next.js)

```bash
# Install Node dependencies
npm install

# Development
npm run dev
# Open http://localhost:3000

# Production build
npm run build
npm start
```

## Deploy to Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/ak-sh1/fraudgraph)

Or manually:
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

The Next.js app loads precomputed demo data from `data/demo/alerts.json` — no Python runtime needed.

## Demo Path (2 minutes)

1. **Queue view**: See ranked alerts by fraud score
2. **Case detail**: Click any alert to see:
   - Transaction timeline
   - Top reason features (SHAP)
   - Ego-graph visualization (shared accounts/merchants/devices)
   - Disposition actions (confirm fraud / false positive)
3. **Metrics**: Check `reports/EVAL.md` for Precision@K and PR-AUC

## Repository Structure

```
fraudgraph/
  src/              # Python pipeline
    generate/       # Synthetic data generator
    features/       # Graph feature engineering
    models/         # Rules + tabular models
    eval/           # Precision@K, PR-AUC harness
  app/              # Next.js case queue UI
  data/
    demo/           # Committed sample for cold demo
  reports/          # EVAL.md, metrics.json
  notebooks/        # EDA
```

## What This Does NOT Claim

- **Not real bank data**: Synthetic fraud patterns only
- **No production guarantees**: This is a portfolio demo, not a deployed system
- **Graph features are relational signals**, not proof of organized crime
- **No GNN**: NetworkX features only (degree, shared neighbors, velocity)
- **No real-time streaming**: Offline batch evaluation

## Tech Stack

- **Data**: Python, pandas, NetworkX
- **Models**: scikit-learn, LightGBM, SHAP
- **UI**: Next.js 14 (App Router), React, D3.js
- **Deploy**: Vercel

## License

MIT

---

Built by Akash Gupta • [GitHub](https://github.com/ak-sh1)
