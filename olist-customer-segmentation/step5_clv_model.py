# =============================================================================
# STEP 5 — CLV Prediction Model  (Random Forest)
# HOW TO RUN:  python step5_clv_model.py
# WHAT IT DOES: Trains a Random Forest model to predict Customer Lifetime
#               Value, compares it against 3 other models, saves the best
#               model and the final dataset with CLV predictions
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LinearRegression, Ridge
from sklearn.ensemble      import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics       import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
os.makedirs("outputs/clv", exist_ok=True)
os.makedirs("models",      exist_ok=True)

# Styling
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor":   "white",
    "axes.spines.top":  False,   "axes.spines.right": False,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,      "axes.labelsize":    11,
    "xtick.labelsize":  10,      "ytick.labelsize":   10,
})
SEG_COLORS = {
    "Champions":"#1D9E75","Loyal":"#378ADD",
    "At-Risk":"#EF9F27","New":"#D4537E","Hibernating":"#888780",
}
SEG_ORDER = ["Champions","Loyal","At-Risk","New","Hibernating"]

print("=" * 60)
print("  STEP 5 — CLV Prediction Model")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
print("\n[1/7] Loading segmented customer data...")
df = pd.read_csv("data/customer_segments.csv")
print(f"     Rows: {len(df):,}   Columns: {list(df.columns)}")

# ─────────────────────────────────────────────────────────────
# BUILD CLV TARGET VARIABLE
# ─────────────────────────────────────────────────────────────
print("\n[2/7] Building CLV target variable...")
# Formula: total spend * log(orders) * margin
# Simple but realistic — used in many industry projects
df["CLV"] = (
    df["Monetary"] *
    np.log1p(df["Frequency"]) *
    1.15                          # assumed 15% gross margin
).round(2)

# Remove top 1% extreme outliers from target
clv_99    = df["CLV"].quantile(0.99)
df["CLV"] = df["CLV"].clip(upper=clv_99)

print(f"     CLV  min   : R${df['CLV'].min():.2f}")
print(f"     CLV  mean  : R${df['CLV'].mean():.2f}")
print(f"     CLV  median: R${df['CLV'].median():.2f}")
print(f"     CLV  max   : R${df['CLV'].max():.2f}")

# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n[3/7] Engineering features...")

# Encode segment as a number (Champions=4 best, Hibernating=0 worst)
seg_ordinal = {
    "Champions":4,"Loyal":3,"At-Risk":2,"New":1,"Hibernating":0
}
df["Segment_Encoded"] = df["Segment"].map(seg_ordinal).fillna(0)

# Average order value
df["Avg_Order_Value"] = (df["Monetary"] / df["Frequency"]).round(2)

# Log transforms to reduce skewness
df["Log_Monetary"]  = np.log1p(df["Monetary"])
df["Log_Frequency"] = np.log1p(df["Frequency"])
df["Log_Recency"]   = np.log1p(df["Recency"])

FEATURE_COLS = [
    "Recency", "Frequency", "Monetary",
    "Log_Monetary", "Log_Frequency", "Log_Recency",
    "Avg_Order_Value", "Cluster", "Segment_Encoded"
]

X = df[FEATURE_COLS].fillna(0)
y = df["CLV"]
print(f"     Features : {FEATURE_COLS}")
print(f"     Samples  : {len(X):,}")

# ─────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
print("\n[4/7] Splitting data  (80% train / 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"     Train: {len(X_train):,}   Test: {len(X_test):,}")

# ─────────────────────────────────────────────────────────────
# TRAIN 4 MODELS AND COMPARE
# ─────────────────────────────────────────────────────────────
print("\n[5/7] Training and comparing 4 models...")
models_to_try = {
    "Linear Regression":
        LinearRegression(),
    "Ridge Regression":
        Ridge(alpha=1.0),
    "Random Forest":
        RandomForestRegressor(n_estimators=200, max_depth=10,
                              random_state=42, n_jobs=-1),
    "Gradient Boosting":
        GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                  learning_rate=0.05, random_state=42),
}

results = {}
for name, model in models_to_try.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)
    results[name] = {"model":model, "preds":preds,
                     "MAE":mae, "RMSE":rmse, "R2":r2}
    print(f"     {name:<25} | MAE=R${mae:>7.2f} | "
          f"RMSE=R${rmse:>8.2f} | R²={r2:.4f}")

best_name = max(results, key=lambda k: results[k]["R2"])
best      = results[best_name]
print(f"\n     ✅  Best model: {best_name}   (R²={best['R2']:.4f})")

# ─────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────
print("\n[6/7] Drawing charts...")

# Chart 1 — Feature Importance (Random Forest)
rf_model = results["Random Forest"]["model"]
fi = pd.Series(rf_model.feature_importances_,
               index=FEATURE_COLS).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = ["#1D9E75" if v > 0.08 else "#B4B2A9" for v in fi.values]
ax.barh(fi.index, fi.values, color=bar_colors, alpha=0.85)
ax.axvline(0.08, color="#EF9F27", linestyle="--", alpha=0.5,
           label="8% threshold")
ax.set_title("Random Forest — Feature Importance for CLV")
ax.set_xlabel("Importance Score")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/clv/01_feature_importance.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("     Saved → outputs/clv/01_feature_importance.png")

# Chart 2 — Actual vs Predicted  +  Model Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

idx    = np.random.choice(len(y_test), min(2000, len(y_test)), replace=False)
act_s  = y_test.values[idx]
pred_s = best["preds"][idx]

axes[0].scatter(act_s, pred_s, alpha=0.3, s=10, color="#1D9E75")
lim = [min(act_s.min(), pred_s.min()), max(act_s.max(), pred_s.max())]
axes[0].plot(lim, lim, "r--", linewidth=1.5, label="Perfect fit")
axes[0].set_xlabel("Actual CLV (R$)")
axes[0].set_ylabel("Predicted CLV (R$)")
axes[0].set_title(f"{best_name}\nActual vs Predicted  (R²={best['R2']:.3f})")
axes[0].legend()

model_names = list(results.keys())
r2_vals     = [results[m]["R2"] for m in model_names]
bar_c       = ["#1D9E75" if m == best_name else "#B4B2A9"
               for m in model_names]
bars = axes[1].barh(model_names, r2_vals, color=bar_c, alpha=0.85)
for bar, val in zip(bars, r2_vals):
    axes[1].text(bar.get_width() + 0.005,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=9)
axes[1].set_xlabel("R² Score")
axes[1].set_title("Model Comparison  (R² — higher is better)")
axes[1].set_xlim(0, 1.1)
plt.tight_layout()
plt.savefig("outputs/clv/02_actual_vs_predicted.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("     Saved → outputs/clv/02_actual_vs_predicted.png")

# Chart 3 — CLV Distribution per Segment
best_model_obj  = results[best_name]["model"]
df["Predicted_CLV"] = best_model_obj.predict(X).round(2)

order_list  = [s for s in SEG_ORDER if s in df["Segment"].values]
colors_list = [SEG_COLORS[s] for s in order_list]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

avg_clv = df.groupby("Segment")["Predicted_CLV"].mean().reindex(order_list)
axes[0].bar(order_list, avg_clv.values, color=colors_list,
            alpha=0.85, edgecolor="white")
axes[0].set_title("Average Predicted CLV per Segment")
axes[0].set_ylabel("Avg CLV (R$)")
for i, val in enumerate(avg_clv.values):
    axes[0].text(i, val + 2, f"R${val:.0f}", ha="center", fontsize=9)

box_data = [df[df["Segment"]==s]["Predicted_CLV"].values
            for s in order_list]
bp = axes[1].boxplot(box_data, patch_artist=True,
                     medianprops=dict(color="white", linewidth=2))
for patch, color in zip(bp["boxes"], colors_list):
    patch.set_facecolor(color); patch.set_alpha(0.75)
axes[1].set_xticklabels(order_list, rotation=15)
axes[1].set_title("CLV Distribution per Segment")
axes[1].set_ylabel("Predicted CLV (R$)")

plt.suptitle("Customer Lifetime Value by Segment", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/clv/03_clv_by_segment.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("     Saved → outputs/clv/03_clv_by_segment.png")

# Chart 4 — Total Revenue Opportunity
fig, ax = plt.subplots(figsize=(9, 5))
total_clv = df.groupby("Segment")["Predicted_CLV"] \
              .sum().reindex(order_list)
bars = ax.bar(order_list, total_clv.values / 1_000_000,
              color=colors_list, alpha=0.85, edgecolor="white")
for bar, val in zip(bars, total_clv.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"R${val/1e6:.2f}M", ha="center", fontsize=9)
ax.set_title("Total Predicted CLV per Segment  (Revenue Opportunity)")
ax.set_ylabel("Total CLV  (R$ millions)")
plt.tight_layout()
plt.savefig("outputs/clv/04_revenue_opportunity.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("     Saved → outputs/clv/04_revenue_opportunity.png")

# ─────────────────────────────────────────────────────────────
# SAVE MODEL + FINAL DATASET
# ─────────────────────────────────────────────────────────────
print("\n[7/7] Saving model and final dataset...")
joblib.dump(best_model_obj, "models/clv_model.pkl")
joblib.dump(FEATURE_COLS,   "models/feature_cols.pkl")
df.to_csv("data/customer_segments_clv.csv", index=False)

print("\n" + "=" * 60)
print("  Final Results")
print("=" * 60)
print(f"  Best model : {best_name}")
print(f"  MAE        : R${best['MAE']:.2f}")
print(f"  RMSE       : R${best['RMSE']:.2f}")
print(f"  R²         : {best['R2']:.4f}")
print("\n  CLV per Segment:")
print(df.groupby("Segment")["Predicted_CLV"]
      .agg(["mean","sum","count"])
      .reindex(order_list).round(2).to_string())
print(f"\n  Total portfolio CLV: R${df['Predicted_CLV'].sum():,.2f}")
print("=" * 60)
print("\n✅  Model training complete!")
print("    data/customer_segments_clv.csv")
print("    models/clv_model.pkl")
print("    models/feature_cols.pkl")
print("    outputs/clv/  (4 charts)")
print("\n    Next → run:  streamlit run app.py")