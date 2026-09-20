# FraudGraph Quick Start

## 2-Minute Demo Path

### Option 1: View the Live Demo (Fastest)

The repository is ready to deploy to Vercel:

1. Go to [vercel.com](https://vercel.com) and sign in
2. Click "Import Project"
3. Import from GitHub: `ak-sh1/fraudgraph`
4. Deploy (no environment variables needed)
5. Open the deployed URL

### Option 2: Local Demo

```bash
# Clone repo
git clone https://github.com/ak-sh1/fraudgraph.git
cd fraudgraph

# View precomputed demo (no Python needed)
npm install
npm run dev
# Open http://localhost:3000
```

The UI loads from precomputed `public/data/alerts.json` — no backend required!

### Option 3: Full Regeneration (Requires Python)

```bash
# 1. Generate data & train models
pip install -r requirements.txt
python3 -m src.generate.synthetic
python3 -m src.features.build
python3 -m src.models.train
python3 -m src.eval.evaluate
python3 -m src.export_demo

# 2. Copy to public
cp data/demo/*.json public/data/

# 3. Run UI
npm install
npm run dev
```

## What You'll See

### Alert Queue (Home)
- 200 alerts ranked by fraud score
- Click any row to see details

### Alert Detail
- Transaction info (amount, account, merchant, device)
- Top fraud reasons (velocity, new device, shared rings, etc.)
- Ego-graph visualization (D3.js force-directed graph)
- Disposition buttons (confirm fraud / false positive)

## Key Metrics

From `reports/EVAL.md`:
- **LightGBM + Graph**: 0.998 PR-AUC, 100% P@200
- **Graph features lift**: +16-28% over tabular-only
- **Top patterns caught**: ATO, card testing, shared-device rings, merchant concentration

## Deploy to Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/ak-sh1/fraudgraph)

Or manually:
```bash
npm i -g vercel
vercel --prod
```

## Project Structure

```
fraudgraph/
  src/              # Python pipeline
    generate/       # Synthetic data
    features/       # Graph features
    models/         # Rules + ML
    eval/           # Metrics
  app/              # Next.js UI
  data/demo/        # Precomputed alerts
  reports/          # EVAL.md + charts
```

## Tech Stack

- **Data**: Python, pandas, NetworkX
- **ML**: scikit-learn, LightGBM
- **UI**: Next.js 14, React, D3.js
- **Deploy**: Vercel

## Limitations

- Synthetic data only (no real fraud)
- Offline evaluation (no streaming)
- No GNN (NetworkX features only)
- Portfolio demo, not production-ready

---

Questions? See README.md or PLAN.md for full details.
