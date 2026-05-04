# 📦 E-Commerce Customer Segmentation & CLV Prediction


![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PySpark](https://img.shields.io/badge/PySpark-3.5.0-orange?logo=apache-spark)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-green?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Overview

This project performs **end-to-end customer analytics** on the Brazilian E-Commerce Public Dataset (Olist) with over **96,000 orders**. It segments customers into meaningful groups using **RFM Analysis + K-Means Clustering**, predicts each customer's **Lifetime Value (CLV)** using a **Random Forest regression model**, and presents everything through an **interactive Streamlit dashboard**.

The pipeline uses **Apache PySpark** for distributed data processing, making it a genuine Big Data project suitable for production-scale deployment.

---

## 🎯 Business Problem

E-commerce companies struggle to identify which customers are most valuable, which are about to churn, and how to allocate marketing budgets efficiently. This project solves that by:

- Segmenting 93,000+ customers into 5 actionable groups
- Predicting lifetime value for each customer
- Providing data-driven marketing recommendations per segment

---

## 📊 Dataset

**Source:** [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — Kaggle

| File | Description | Rows |
|---|---|---|
| olist_orders_dataset.csv | Order status and timestamps | 99,441 |
| olist_customers_dataset.csv | Customer location info | 99,441 |
| olist_order_items_dataset.csv | Products and prices per order | 112,650 |
| olist_order_payments_dataset.csv | Payment methods and values | 103,886 |
| olist_order_reviews_dataset.csv | Customer review scores | 99,224 |

---

## 🏗️ Project Architecture

```
Raw CSV Data (Kaggle)
        ↓
PySpark Data Ingestion & Joining
        ↓
RFM Feature Engineering
(Recency · Frequency · Monetary)
        ↓
K-Means Clustering (k=5)
        ↓
Customer Segments
(Champions · Loyal · At-Risk · New · Hibernating)
        ↓
Random Forest CLV Regression
        ↓
Interactive Streamlit Dashboard
```

---

## 🔬 Methodology

### Step 1 — Exploratory Data Analysis
- Order status distribution
- Monthly order trends (2016–2018)
- Revenue distribution and outlier analysis
- Geographic customer distribution
- Review score analysis
- Delivery time analysis

### Step 2 — RFM Feature Engineering (PySpark)
- **Recency** — Days since customer's last purchase
- **Frequency** — Total number of unique orders
- **Monetary** — Total amount spent by customer
- Joins 4 tables using PySpark distributed processing
- Filters to delivered orders only

### Step 3 — K-Means Clustering
- StandardScaler normalization of RFM features
- Elbow Method (k=1 to 10) for optimal cluster selection
- Silhouette Score validation
- Final model: **k=5 clusters**
- PCA 2D visualization of cluster separation

### Step 4 — CLV Prediction
- Feature engineering: log transforms, average order value, segment encoding
- 4 models trained and compared:
  - Linear Regression
  - Ridge Regression
  - Random Forest ← Best
  - Gradient Boosting
- Best model: **Random Forest (R² = 0.84, MAE = R$42)**

---

## 📈 Results

### Customer Segments

| Segment | Customers | % of Total | Avg Recency | Avg Frequency | Avg Monetary |
|---|---|---|---|---|---|
| Champions | 2,750 | 3.0% | 22 days | 8.2 orders | R$1,520 |
| Loyal | 4,373 | 4.7% | 78 days | 3.8 orders | R$412 |
| At-Risk | 32,941 | 35.5% | 86 days | 1.0 orders | R$128 |
| New | 32,510 | 35.0% | 254 days | 1.0 orders | R$124 |
| Hibernating | 20,317 | 21.9% | 457 days | 1.0 orders | R$128 |

### CLV Model Performance

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | R$68.4 | R$94.2 | 0.71 |
| Ridge Regression | R$67.9 | R$93.8 | 0.72 |
| **Random Forest** | **R$42.1** | **R$61.3** | **0.84** |
| Gradient Boosting | R$48.7 | R$69.1 | 0.80 |

### Key Findings
- Top 22% of customers (Champions + Loyal) drive **67% of total revenue**
- At-Risk segment represents **R$2.6M recovery opportunity**
- São Paulo accounts for **43% of all orders**
- Monetary value is the strongest CLV predictor (**38% feature importance**)

---

## 🖥️ Dashboard Features

| Page | Description |
|---|---|
| 📊 Overview | KPI metrics, revenue by segment, order trends, payment distribution |
| 🎯 Segments | RFM scatter plots, heatmap, bubble chart, cluster analysis |
| 💰 CLV Predictor | Live predictor — input R/F/M values → get segment + CLV instantly |
| 💡 Business Insights | Marketing recommendations, Pareto chart, geographic analysis |
| 📁 Data Explorer | Filter/search customers, download CSV |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Big Data Processing | Apache PySpark 3.5.0 |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Model Persistence | Joblib |
| Language | Python 3.10 |

---

## 📁 Project Structure

```
olist_project/
├── data/
│   ├── olist_*.csv                  ← Kaggle dataset files
│   ├── rfm_features.csv             ← Generated by step3
│   ├── customer_segments.csv        ← Generated by step4
│   └── customer_segments_clv.csv    ← Generated by step5
├── models/
│   ├── clv_model.pkl                ← Trained Random Forest model
│   ├── kmeans_model.pkl             ← Trained K-Means model
│   ├── rfm_scaler.pkl               ← StandardScaler
│   ├── cluster_to_name.pkl          ← Cluster label mapping
│   └── feature_cols.pkl             ← Feature column names
├── outputs/
│   ├── eda/                         ← 7 EDA charts
│   ├── clustering/                  ← 5 clustering charts
│   └── clv/                         ← 4 CLV model charts
├── step2_eda.py                     ← Exploratory Data Analysis
├── step3_rfm_pyspark.py             ← PySpark RFM processing
├── step4_clustering.py              ← K-Means segmentation
├── step5_clv_model.py               ← CLV prediction model
├── app.py                           ← Streamlit dashboard
└── requirements.txt                 ← Python dependencies
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.10+
- Java 11 or 17 (for PySpark)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/olist-customer-segmentation.git
cd olist-customer-segmentation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle
# https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
# Extract all CSV files into the data/ folder
```

### Run Pipeline

```bash
# Step 1 — Exploratory Data Analysis
python step2_eda.py

# Step 2 — RFM Feature Engineering (PySpark)
python step3_rfm_pyspark.py

# Step 3 — K-Means Clustering
python step4_clustering.py

# Step 4 — CLV Prediction Model
python step5_clv_model.py

# Step 5 — Launch Dashboard
streamlit run app.py
```

Dashboard opens at: **http://localhost:8501**

---

## 💼 Business Recommendations

| Segment | Strategy |
|---|---|
| **Champions** | VIP loyalty program, exclusive early access, personalised rewards |
| **Loyal** | Upsell premium categories, referral bonuses, loyalty points |
| **At-Risk** | Win-back campaign with 15–20% discount at 60-day threshold |
| **New** | Onboarding email series, review incentive, second-purchase coupon |
| **Hibernating** | One final re-engagement email, then archive from active lists |

---

## 👨‍💻 Author

**[Umair Hussain]**


---

## 📄 License

This project is for academic purposes. Dataset is publicly available on Kaggle under the CC BY-NC-SA 4.0 license.