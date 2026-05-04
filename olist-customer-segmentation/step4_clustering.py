# =============================================================================
# STEP 4 — K-Means Customer Segmentation
# HOW TO RUN:  python step4_clustering.py
# WHAT IT DOES: Reads rfm_features.csv, groups customers into 5 segments
#               using K-Means, saves charts and customer_segments.csv
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
os.makedirs("outputs/clustering", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Styling
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor":   "white",
    "axes.spines.top":  False,   "axes.spines.right": False,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,      "axes.labelsize":    11,
    "xtick.labelsize":  10,      "ytick.labelsize":   10,
})
SEG_COLORS = {
    "Champions":   "#1D9E75",
    "Loyal":       "#378ADD",
    "At-Risk":     "#EF9F27",
    "New":         "#D4537E",
    "Hibernating": "#888780",
}
SEG_ORDER = ["Champions", "Loyal", "At-Risk", "New", "Hibernating"]

print("=" * 60)
print("  STEP 4 — K-Means Customer Segmentation")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# LOAD RFM DATA
# ─────────────────────────────────────────────────────────────
print("\n[1/6] Loading RFM features...")
rfm = pd.read_csv("data/rfm_features.csv")
print(f"     Customers loaded: {len(rfm):,}")

# Remove top 0.5% outliers so they don't distort clusters
rfm = rfm[rfm["Monetary"]  <= rfm["Monetary"].quantile(0.995)].copy()
rfm = rfm[rfm["Frequency"] <= 20].copy()
print(f"     After outlier removal: {len(rfm):,}")

# ─────────────────────────────────────────────────────────────
# SCALE RFM  (StandardScaler: mean=0, std=1)
# ─────────────────────────────────────────────────────────────
print("\n[2/6] Scaling RFM features with StandardScaler...")
scaler     = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
print("     Done — each feature now has mean≈0 and std≈1")

# ─────────────────────────────────────────────────────────────
# ELBOW METHOD  — find best k
# ─────────────────────────────────────────────────────────────
print("\n[3/6] Running Elbow Method (k = 1 to 10)...")
wcss       = []
sil_scores = []

for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    km.fit(rfm_scaled)
    wcss.append(km.inertia_)
    if k > 1:
        sil = silhouette_score(rfm_scaled, km.labels_,
                               sample_size=5000, random_state=42)
        sil_scores.append(sil)
        print(f"     k={k:2d}  WCSS={km.inertia_:>12,.0f}"
              f"  Silhouette={sil:.3f}")
    else:
        sil_scores.append(None)
        print(f"     k={k:2d}  WCSS={km.inertia_:>12,.0f}"
              f"  Silhouette= N/A")

# Save elbow + silhouette chart
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ks = list(range(1, 11))

axes[0].plot(ks, wcss, "o-", color="#1D9E75", linewidth=2.5, markersize=7)
axes[0].axvline(5, color="#EF9F27", linestyle="--", alpha=0.7,
                label="k=5  (chosen)")
axes[0].set_title("Elbow Method — WCSS vs k")
axes[0].set_xlabel("Number of Clusters  (k)")
axes[0].set_ylabel("WCSS")
axes[0].legend()

sil_vals = [s for s in sil_scores if s is not None]
axes[1].plot(ks[1:], sil_vals, "o-", color="#378ADD",
             linewidth=2.5, markersize=7)
axes[1].axvline(5, color="#EF9F27", linestyle="--", alpha=0.7,
                label="k=5  (chosen)")
axes[1].set_title("Silhouette Score vs k  (higher = better)")
axes[1].set_xlabel("Number of Clusters  (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].legend()

plt.suptitle("Choosing the Optimal Number of Clusters", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/clustering/01_elbow_silhouette.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("\n     Elbow chart saved → outputs/clustering/01_elbow_silhouette.png")
print("     → k=5 chosen (elbow point + best silhouette)")

# ─────────────────────────────────────────────────────────────
# FIT FINAL K-MEANS  (k=5)
# ─────────────────────────────────────────────────────────────
print("\n[4/6] Fitting final K-Means model  (k=5)...")
km_final         = KMeans(n_clusters=5, random_state=42,
                           n_init=20, max_iter=500)
rfm["Cluster"]   = km_final.fit_predict(rfm_scaled)

# ── Name each cluster based on its RFM centre values ──────────
centers_orig = scaler.inverse_transform(km_final.cluster_centers_)
centers_df   = pd.DataFrame(centers_orig,
                             columns=["Recency", "Frequency", "Monetary"])
centers_df["Cluster"] = range(5)

# Score: lower Recency is better, higher F and M are better
centers_df["Score"] = (
    -centers_df["Recency"]   / (centers_df["Recency"].max()   + 1e-9) * 0.3 +
     centers_df["Frequency"] / (centers_df["Frequency"].max() + 1e-9) * 0.4 +
     centers_df["Monetary"]  / (centers_df["Monetary"].max()  + 1e-9) * 0.3
)

seg_names      = ["Champions", "Loyal", "At-Risk", "New", "Hibernating"]
centers_sorted = centers_df.sort_values("Score", ascending=False) \
                            .reset_index(drop=True)
cluster_to_name = {
    int(row["Cluster"]): seg_names[i]
    for i, row in centers_sorted.iterrows()
}

rfm["Segment"] = rfm["Cluster"].map(cluster_to_name)
print(f"     Cluster → Segment mapping: {cluster_to_name}")

# ─────────────────────────────────────────────────────────────
# SEGMENT ANALYSIS TABLE
# ─────────────────────────────────────────────────────────────
seg_stats = rfm.groupby("Segment").agg(
    Count     =("customer_unique_id", "count"),
    Avg_R     =("Recency",   "mean"),
    Avg_F     =("Frequency", "mean"),
    Avg_M     =("Monetary",  "mean"),
    Total_Rev =("Monetary",  "sum"),
).round(1)
seg_stats["Pct_Customers"] = \
    (seg_stats["Count"] / len(rfm) * 100).round(1)
seg_stats["Pct_Revenue"] = \
    (seg_stats["Total_Rev"] / seg_stats["Total_Rev"].sum() * 100).round(1)

print("\n" + "=" * 60)
print("  Segment Summary")
print("=" * 60)
print(seg_stats[["Count","Pct_Customers","Avg_R","Avg_F",
                  "Avg_M","Pct_Revenue"]].to_string())

# ─────────────────────────────────────────────────────────────
# CHART 2 — RFM Scatter Plots
# ─────────────────────────────────────────────────────────────
print("\n[5/6] Drawing charts...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for seg, grp in rfm.groupby("Segment"):
    color  = SEG_COLORS.get(seg, "#888")
    sample = grp.sample(min(500, len(grp)), random_state=42)
    axes[0].scatter(sample["Recency"], sample["Monetary"],
                    alpha=0.45, s=12, color=color, label=seg)
    axes[1].scatter(sample["Frequency"], sample["Monetary"],
                    alpha=0.45, s=12, color=color, label=seg)

axes[0].set_xlabel("Recency (days)")
axes[0].set_ylabel("Monetary (R$)")
axes[0].set_title("Recency vs Monetary")
axes[0].legend(markerscale=2, fontsize=9)

axes[1].set_xlabel("Frequency (# orders)")
axes[1].set_ylabel("Monetary (R$)")
axes[1].set_title("Frequency vs Monetary")
axes[1].legend(markerscale=2, fontsize=9)

plt.suptitle("RFM Customer Segments", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/clustering/02_rfm_scatter.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("     Saved → outputs/clustering/02_rfm_scatter.png")

# Chart 3 — Segment size and revenue
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
order_list   = [s for s in SEG_ORDER if s in seg_stats.index]
colors_list  = [SEG_COLORS[s] for s in order_list]
seg_reindexed = seg_stats.reindex(order_list)

axes[0].bar(order_list, seg_reindexed["Count"],
            color=colors_list, alpha=0.85, edgecolor="white")
axes[0].set_title("Customer Count per Segment")
axes[0].set_ylabel("Number of Customers")
for i, val in enumerate(seg_reindexed["Count"]):
    axes[0].text(i, val + 150, f"{val:,}", ha="center", fontsize=9)

axes[1].bar(order_list, seg_reindexed["Total_Rev"] / 1000,
            color=colors_list, alpha=0.85, edgecolor="white")
axes[1].set_title("Total Revenue per Segment")
axes[1].set_ylabel("Revenue (R$ thousands)")
for i, val in enumerate(seg_reindexed["Total_Rev"] / 1000):
    axes[1].text(i, val + 30, f"R${val:,.0f}k", ha="center", fontsize=9)

plt.suptitle("Segment Size and Revenue Contribution", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/clustering/03_segment_revenue.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("     Saved → outputs/clustering/03_segment_revenue.png")

# Chart 4 — Snake Plot (normalised RFM per segment)
rfm_norm = rfm.copy()
for c in ["Recency", "Frequency", "Monetary"]:
    mn = rfm_norm[c].min(); mx = rfm_norm[c].max()
    rfm_norm[c] = (rfm_norm[c] - mn) / (mx - mn + 1e-9)
rfm_norm["Recency"] = 1 - rfm_norm["Recency"]   # flip: lower = better

melt = rfm_norm.groupby("Segment")[["Recency","Frequency","Monetary"]] \
               .mean().reindex(order_list)

fig, ax = plt.subplots(figsize=(8, 5))
for seg in order_list:
    if seg in melt.index:
        row = melt.loc[seg]
        ax.plot(["Recency (inverted)","Frequency","Monetary"],
                row.values, "o-",
                color=SEG_COLORS.get(seg, "#888"),
                linewidth=2.5, markersize=8, label=seg)
ax.set_title("RFM Snake Plot  (normalised, higher = better)")
ax.set_ylabel("Normalised Score  (0–1)")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("outputs/clustering/04_snake_plot.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("     Saved → outputs/clustering/04_snake_plot.png")

# Chart 5 — PCA 2D visualisation
pca        = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(rfm_scaled)
rfm["PCA1"] = pca_coords[:, 0]
rfm["PCA2"] = pca_coords[:, 1]

fig, ax = plt.subplots(figsize=(9, 6))
for seg, grp in rfm.groupby("Segment"):
    s = grp.sample(min(800, len(grp)), random_state=42)
    ax.scatter(s["PCA1"], s["PCA2"],
               alpha=0.4, s=10,
               color=SEG_COLORS.get(seg, "#888"), label=seg)
ax.set_title("PCA 2D View of K-Means Clusters")
ax.set_xlabel(f"PCA 1  ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PCA 2  ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.legend(markerscale=2.5, fontsize=9)
plt.tight_layout()
plt.savefig("outputs/clustering/05_pca_clusters.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("     Saved → outputs/clustering/05_pca_clusters.png")

rfm.drop(columns=["PCA1","PCA2"], inplace=True, errors="ignore")

# ─────────────────────────────────────────────────────────────
# SAVE FILES
# ─────────────────────────────────────────────────────────────
print("\n[6/6] Saving files...")
rfm.to_csv("data/customer_segments.csv", index=False)
joblib.dump(scaler,          "models/rfm_scaler.pkl")
joblib.dump(km_final,        "models/kmeans_model.pkl")
joblib.dump(cluster_to_name, "models/cluster_to_name.pkl")

print("\n✅  Clustering complete!")
print("    data/customer_segments.csv")
print("    models/rfm_scaler.pkl")
print("    models/kmeans_model.pkl")
print("    models/cluster_to_name.pkl")
print("    outputs/clustering/  (5 charts)")
print("\n    Next → run:  python step5_clv_model.py")