<div align="center">

<img src="https://img.shields.io/badge/-%F0%9F%A7%A0%20SegmentIQ-0d1117?style=for-the-badge&labelColor=0d1117&color=4f8cf7" height="40"/>

# SegmentIQ — Customer Intelligence Platform

**End-to-end K-Means++ customer segmentation system for financial product recommendation.**
**Built with Flask · scikit-learn · Neon PostgreSQL · Deployed on Vercel.**

<br/>

[![Live Demo](https://img.shields.io/badge/▲%20Live%20Demo-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://segment-iq-customer-segmentation-in.vercel.app)
[![GitHub](https://img.shields.io/badge/GitHub-Hassan141998-181717?style=for-the-badge&logo=github)](https://github.com/Hassan141998)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Hassan%20Ahmed-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hassan-ahmed-98304030a/)
[![Upwork](https://img.shields.io/badge/Upwork-Available%20for%20Hire-6FDA44?style=for-the-badge&logo=upwork&logoColor=white)](https://www.upwork.com)

<br/>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)
![Neon](https://img.shields.io/badge/Neon-PostgreSQL-00E599?logo=postgresql&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-Deployed-000000?logo=vercel)
![License](https://img.shields.io/badge/License-MIT-22C55E)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [ML Pipeline](#-ml-pipeline)
- [Dashboard Sections](#-dashboard-sections)
- [API Reference](#-api-reference)
- [Customer Segments](#-customer-segments)
- [Tech Stack](#-tech-stack)
- [Local Setup](#️-local-setup)
- [Deploy to Vercel + Neon](#️-deploy-to-vercel--neon)
- [Model Results](#-model-results)
- [Author](#-author)

---

## 🧠 Overview

**SegmentIQ** is a production-grade machine learning system that segments financial customers into distinct behavioral groups using **K-Means++ clustering**. It automatically selects the optimal number of clusters using a composite scoring system, profiles each segment, and recommends personalized financial products in real time.

The project demonstrates a complete data science workflow — from synthetic data generation and feature engineering through model training, evaluation, and deployment — all wrapped in a polished, interactive web dashboard.

> Built as a portfolio project showcasing ML engineering, full-stack development, cloud deployment, and database integration skills.

---

## 🌐 Live Demo

**[▲ segment-iq-customer-segmentation-in.vercel.app](https://segment-iq-customer-segmentation-in.vercel.app)**

| Section | What you can do |
|---|---|
| **Overview** | View KPI metrics, cluster distribution, 2D PCA scatter, multi-metric evaluation |
| **Segments** | Explore financial profiles and product recommendations per segment |
| **Analytics** | Interact with 3D scatter, radar chart, feature importance, elbow curve |
| **Predict** | Enter customer data, get segment classification + recommendations instantly |
| **Customers** | Browse 1,000 segmented customers, filter by segment, export to CSV |
| **History** | See last 20 real-time predictions stored in Neon PostgreSQL |
| **Settings** | Retrain the model and view run history |

---

## ✨ Features

### Machine Learning Pipeline
- **K-Means++** clustering with `n_init=20` for stable centroid initialization
- **Automatic optimal K selection** via composite metric scoring:
  - Silhouette Score (40% weight)
  - Calinski-Harabász Index (30% weight)
  - Davies-Bouldin Index (30% weight)
- **RobustScaler** preprocessing — resistant to financial data outliers
- **PCA** dimensionality reduction for 2D and visualization (~74% variance retained)
- **Feature engineering** — 5 derived financial ratios from 11 raw features
- Full evaluation suite: Silhouette · Davies-Bouldin · Calinski-Harabász · Elbow/Inertia

### Dashboard & Visualization
- 7-section single-page application (SPA) — no page reloads
- Interactive **Plotly.js** charts: 2D scatter, 3D scatter, radar, bar, multi-line metrics, elbow
- **Dark professional UI** — Space Grotesk + JetBrains Mono typography
- Segment profile cards with avg income, age, credit score, savings, and product recommendations
- Real-time **customer classifier** form with instant API response
- Paginated, filterable customer table with CSV export

### Infrastructure & Deployment
- **Vercel-optimized architecture**: model trained at build time, results served as static JSON — zero cold start latency for normal requests
- **Neon PostgreSQL** persistence: customers, predictions, and model runs stored serverlessly
- **Graceful degradation**: runs in local/in-memory mode when no `DATABASE_URL` is set
- 14 REST API endpoints
- `python-dotenv` for local environment management

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      VERCEL BUILD TIME                      │
│  build.py → ml/pipeline.py → static/precomputed.json       │
│  (train model, serialize results — happens once per deploy) │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                     VERCEL RUNTIME                          │
│                                                             │
│  Browser ──HTTP──► Flask (app.py)                          │
│                         │                                   │
│            ┌────────────┼────────────┐                      │
│            │            │            │                      │
│      Read JSON     sklearn lazy   Neon DB                   │
│   (instant, 248KB) (predict only) (persist)                 │
└─────────────────────────────────────────────────────────────┘
```

**Key design decision:** scikit-learn (270MB) is a build-time dependency only. At runtime, the Flask app reads a pre-trained 248KB JSON file — making every page load instant. sklearn is only imported lazily for `/api/predict` and `/api/retrain`.

---

## 🗂️ Project Structure

```
segmentiq/
│
├── app.py                    # Flask application — Vercel entry point (14 API routes)
├── build.py                  # Build-time script: trains model → static/precomputed.json
├── database.py               # Neon PostgreSQL integration (psycopg2)
│
├── ml/
│   ├── __init__.py
│   └── pipeline.py           # Full ML pipeline:
│                             #   generate_data() → engineer_features() → train()
│                             #   → evaluate → build_results() → predict()
│
├── static/
│   └── precomputed.json      # Pre-trained model results (248KB, committed to repo)
│
├── templates/
│   └── dashboard.html        # Single-file SPA: 7 sections, Plotly.js charts
│
├── vercel.json               # Vercel deployment: buildCommand + routes
├── requirements.txt          # flask · scikit-learn · pandas · numpy · psycopg2 · dotenv
├── .env.example              # Template for local DATABASE_URL setup
├── .gitignore
└── README.md
```

---

## 🤖 ML Pipeline

### 1. Data Generation
Generates 1,000 synthetic customers across 6 realistic financial archetypes with controlled statistical distributions:

```python
from ml.pipeline import generate_data
df = generate_data(seed=42)
# 1,000 rows × 11 features
```

### 2. Feature Engineering
Derives 5 financial ratio features from raw inputs:

| Engineered Feature | Formula |
|---|---|
| `savings_to_income_ratio` | savings_balance / annual_income |
| `investment_to_income_ratio` | investment_amount / annual_income |
| `loan_to_income_ratio` | loan_amount / annual_income |
| `wealth_index` | (savings + investment − loan) / income |
| `engagement_activity_score` | 0.4×digital + 0.4×transactions + 0.2×products×5 |

### 3. Preprocessing
```python
scaler = RobustScaler()          # handles financial outliers
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)        # for visualization
X_pca = pca.fit_transform(X_scaled)
```

### 4. Optimal K Selection
Composite scoring across K = 2…9:

```
Composite Score = 0.40 × silhouette_norm
               + 0.30 × calinski_harabasz_norm
               + 0.30 × (1 − davies_bouldin_norm)
```

### 5. Training
```python
model = KMeans(n_clusters=optimal_k, init="k-means++", n_init=20, random_state=42)
model.fit(X_scaled)
```

### 6. Real-time Prediction
```python
from ml.pipeline import predict
result = predict({
    "age": 42, "annual_income": 95000, "credit_score": 740,
    "savings_balance": 45000, "investment_amount": 30000, ...
})
# → {"cluster": 0, "segment_name": "High-Value Investors", "recommendations": [...]}
```

---

## 📊 Dashboard Sections

| Section | Charts / Features |
|---|---|
| **Overview** | 4 KPI cards · Cluster distribution bar chart · 2D PCA scatter · Multi-metric line chart (Silhouette, DB, CH, Inertia) |
| **Segments** | Profile cards per cluster — size, avg income, age, savings, credit score, color-coded + product recommendation tags |
| **Analytics** | Interactive 3D scatter (Income × Savings × Investment) · Radar overlay chart · Feature importance horizontal bar · Elbow curve with fill |
| **Predict** | 11-field customer form → POST `/api/predict` → segment name + color + recommendation list |
| **Customers** | Paginated table (50/page) · Cluster filter dropdown · Colored segment badges · CSV export |
| **History** | Prediction log from Neon DB — age, income, credit, segment, timestamp |
| **Settings** | Retrain button (POST `/api/retrain`) · Model run history table · System info grid |

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/summary` | KPI metrics + all cluster profiles |
| `GET` | `/api/scatter2d` | PCA 2D coordinates per cluster |
| `GET` | `/api/scatter3d` | Income × Savings × Investment 3D data |
| `GET` | `/api/radar` | Normalized radar chart feature data |
| `GET` | `/api/elbow` | Inertia values for K = 2…10 |
| `GET` | `/api/metrics` | All evaluation metrics per K value |
| `GET` | `/api/feature_importance` | Between/within variance ratio scores |
| `POST` | `/api/predict` | Classify a new customer into a segment |
| `POST` | `/api/retrain` | Re-run full training pipeline |
| `GET` | `/api/customers` | Paginated + filterable customer list |
| `GET` | `/api/history` | Last 20 predictions (Neon DB) |
| `GET` | `/api/model_runs` | Training run history (Neon DB) |
| `GET` | `/api/export` | Download segmented customers as CSV |
| `GET` | `/api/health` | Status: DB connection, active K, data source |

### Example: Classify a customer

```bash
curl -X POST https://segment-iq-customer-segmentation-in.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 42,
    "annual_income": 95000,
    "savings_balance": 45000,
    "credit_score": 740,
    "monthly_transactions": 55,
    "loan_amount": 120000,
    "investment_amount": 30000,
    "risk_appetite_score": 3.2,
    "years_as_customer": 8,
    "num_products": 4,
    "digital_engagement_score": 72
  }'
```

**Response:**
```json
{
  "cluster": 0,
  "segment_name": "High-Value Investors",
  "color": "#FF6B6B",
  "recommendations": [
    "Premium Investment Portfolio",
    "Wealth Management Advisory",
    "Private Banking",
    "Tax Optimization Services"
  ]
}
```

---

## 🎯 Customer Segments

| Cluster | Segment | Typical Profile | Recommended Products |
|---|---|---|---|
| **0** | High-Value Investors | Age 45+ · $120K income · $80K savings · Credit 780 · Low risk | Wealth Management · Private Banking · Tax Optimization |
| **1** | Young Professionals | Age 28 · $65K income · High transactions · Digital-first | Growth Fund · Digital Banking · Career Insurance |
| **2** | Conservative Retirees | Age 65 · Fixed income · $120K savings · Very low risk | Fixed Bonds · Annuities · Healthcare Insurance |
| **3** | Middle-Class Families | Age 40 · $75K income · Mortgage-focused · Moderate risk | Life Insurance · Education Savings · Home Mortgage |
| **4** | Budget-Conscious Savers | Age 35 · $30K income · Careful spenders · Low products | High-Yield Savings · Secured Credit Card · Micro-Investment |
| **5** | High Spenders | Age 33 · $90K income · Low savings · 120 txns/month | Cashback Cards · BNPL · Lifestyle Insurance · Rewards |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **ML** | scikit-learn 1.4 | K-Means++, RobustScaler, PCA, metrics |
| **Data** | pandas 2.1 · NumPy 1.26 | Feature engineering, data manipulation |
| **Backend** | Flask 3.0 | REST API, routing, templating |
| **Database** | Neon PostgreSQL (psycopg2) | Persist customers, predictions, model runs |
| **Frontend** | Vanilla JS · Plotly.js 2.27 | Interactive charts, SPA navigation |
| **Fonts** | Space Grotesk · JetBrains Mono | Professional dark UI typography |
| **Persistence** | joblib | Model serialization (local dev) |
| **Env** | python-dotenv | Local `.env` loading |
| **Deployment** | Vercel | Serverless Flask hosting |
| **CI/CD** | GitHub → Vercel | Auto-deploy on every `git push` |

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.9+
- pip

### 1. Clone
```bash
git clone https://github.com/Hassan141998/SegmentIQ-Customer-Segmentation-Intelligence-Platform.git
cd SegmentIQ-Customer-Segmentation-Intelligence-Platform
```

### 2. Virtual environment
```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\activate
```
```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment variables (optional — for Neon DB)
```bash
cp .env.example .env
# Edit .env and paste your Neon DATABASE_URL
```

### 5. Build and run
```bash
# Step 1: train model and generate precomputed.json
python build.py

# Step 2: start Flask
python app.py
```

Open **http://127.0.0.1:5000** 🚀

> The app works without a `DATABASE_URL` — it shows a yellow "Local Mode" badge and uses in-memory data only.

---

## ☁️ Deploy to Vercel + Neon

### Step 1 — Create Neon Database
1. Go to **[neon.tech](https://neon.tech)** → Sign up → **New Project**
2. Copy your **Connection String** (`postgresql://...`)
3. Tables are auto-created on first request via `init_db()` — no migrations needed

### Step 2 — Push to GitHub
```powershell
# Windows PowerShell — run each line separately
git init
git add .
git commit -m "feat: SegmentIQ v3"
git branch -M main
git remote add origin https://github.com/Hassan141998/SegmentIQ-Customer-Segmentation-Intelligence-Platform.git
git push -u origin main
```

### Step 3 — Deploy on Vercel
1. Go to **[vercel.com](https://vercel.com)** → **Add New Project**
2. Import your GitHub repository
3. Framework Preset: **Other**
4. Add environment variable:
   ```
   DATABASE_URL = postgresql://neondb_owner:...@ep-xxxx.neon.tech/neondb?sslmode=require
   ```
5. Click **Deploy** ✅

### What happens at build time
```
🔧 SegmentIQ build: training model...
✅ Precomputed results saved → static/precomputed.json (248 KB)
   Clusters: 2 | Silhouette: 0.556 | Customers: 1,000
```

Every `git push` triggers an automatic redeploy.

### Step 4 — Seed Neon DB
After first deploy, visit `/api/retrain` (POST) once to seed your Neon DB with all 1,000 customers.

---

## 📈 Model Results

| Metric | Value | Interpretation |
|---|---|---|
| **Silhouette Score** | `0.556` | Good cluster separation (range: −1 to 1) |
| **Davies-Bouldin Index** | `0.702` | Compact clusters (lower = better) |
| **Calinski-Harabász Score** | `1048` | Well-defined clusters (higher = better) |
| **PCA Explained Variance** | `74.1%` | 2 components capture most information |
| **Dataset** | 1,000 customers | 11 raw + 5 engineered = 16 total features |
| **Optimal K** | Auto-selected | Composite scoring across K = 2…9 |

---

## 🗃️ Database Schema (Neon PostgreSQL)

```sql
-- Segmented customer records
CREATE TABLE customers (
    id                       SERIAL PRIMARY KEY,
    customer_id              TEXT UNIQUE,
    age                      INTEGER,
    annual_income            NUMERIC,
    savings_balance          NUMERIC,
    credit_score             INTEGER,
    monthly_transactions     INTEGER,
    loan_amount              NUMERIC,
    investment_amount        NUMERIC,
    risk_appetite_score      NUMERIC,
    years_as_customer        INTEGER,
    num_products             INTEGER,
    digital_engagement_score NUMERIC,
    cluster                  INTEGER,
    segment_name             TEXT,
    created_at               TIMESTAMPTZ DEFAULT NOW()
);

-- Real-time prediction log
CREATE TABLE predictions (
    id              SERIAL PRIMARY KEY,
    age             INTEGER,
    annual_income   NUMERIC,
    credit_score    INTEGER,
    savings_balance NUMERIC,
    cluster         INTEGER,
    segment_name    TEXT,
    recommendations JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Model training history
CREATE TABLE model_runs (
    id                  SERIAL PRIMARY KEY,
    optimal_k           INTEGER,
    silhouette_score    NUMERIC,
    davies_bouldin      NUMERIC,
    calinski_harabasz   NUMERIC,
    n_customers         INTEGER,
    run_at              TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: description"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

<div align="center">

### Hassan Ahmed
**Data Scientist & ML Engineer**

[![GitHub](https://img.shields.io/badge/GitHub-Hassan141998-181717?style=for-the-badge&logo=github)](https://github.com/Hassan141998)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Hassan%20Ahmed-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hassan-ahmed-98304030a/)
[![Upwork](https://img.shields.io/badge/Upwork-Hire%20Me-6FDA44?style=for-the-badge&logo=upwork&logoColor=white)](https://www.upwork.com)

**Available for freelance projects in:**
Machine Learning · Data Science · Python Backend · Dashboard Development · MLOps

</div>

---

<div align="center">

*Built with Python · scikit-learn · Flask · Neon PostgreSQL · Vercel*

**⭐ Star this repo if you found it useful!**

</div>
