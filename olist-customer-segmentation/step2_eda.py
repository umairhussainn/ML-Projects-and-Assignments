# =============================================================================
# STEP 2 — Exploratory Data Analysis (EDA)
# HOW TO RUN:  python step2_eda.py
# WHAT IT DOES: Reads the Kaggle CSV files and saves 7 charts to outputs/eda/
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Create output folders if they don't exist
os.makedirs("outputs/eda", exist_ok=True)

# Styling for all charts
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
})
COLORS = ["#1D9E75", "#378ADD", "#EF9F27", "#D4537E", "#888780"]

print("=" * 60)
print("  STEP 2 — Exploratory Data Analysis")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
print("\nLoading CSV files from data/ folder...")

orders    = pd.read_csv("data/olist_orders_dataset.csv",
                        parse_dates=["order_purchase_timestamp",
                                     "order_delivered_customer_date"])
customers = pd.read_csv("data/olist_customers_dataset.csv")
items     = pd.read_csv("data/olist_order_items_dataset.csv")
payments  = pd.read_csv("data/olist_order_payments_dataset.csv")
reviews   = pd.read_csv("data/olist_order_reviews_dataset.csv")

print(f"  Orders    : {orders.shape[0]:,} rows")
print(f"  Customers : {customers.shape[0]:,} rows")
print(f"  Items     : {items.shape[0]:,} rows")
print(f"  Payments  : {payments.shape[0]:,} rows")
print(f"  Reviews   : {reviews.shape[0]:,} rows")

# ─────────────────────────────────────────────────────────────
# CHART 1 — Order Status Distribution
# ─────────────────────────────────────────────────────────────
print("\n[1/7] Drawing order status chart...")
status_counts = orders["order_status"].value_counts()

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(status_counts.index, status_counts.values,
               color=COLORS[0], alpha=0.85)
for bar, val in zip(bars, status_counts.values):
    ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=9)
ax.set_xlabel("Number of Orders")
ax.set_title("Order Status Distribution")
ax.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig("outputs/eda/01_order_status.png", dpi=150, bbox_inches="tight")
plt.close()
print("     Saved → outputs/eda/01_order_status.png")

# ─────────────────────────────────────────────────────────────
# CHART 2 — Monthly Order Trend
# ─────────────────────────────────────────────────────────────
print("[2/7] Drawing monthly order trend...")
delivered = orders[orders["order_status"] == "delivered"].copy()
delivered["month"] = delivered["order_purchase_timestamp"].dt.to_period("M")
monthly = delivered.groupby("month").size().reset_index(name="count")
monthly["month_str"] = monthly["month"].astype(str)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(monthly["month_str"], monthly["count"],
        color=COLORS[0], linewidth=2.5, marker="o", markersize=4)
ax.fill_between(monthly["month_str"], monthly["count"],
                alpha=0.12, color=COLORS[0])
ax.set_title("Monthly Delivered Orders (2016–2018)")
ax.set_ylabel("Number of Orders")
ax.set_xlabel("Month")
plt.xticks(rotation=45, ha="right")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig("outputs/eda/02_monthly_orders.png", dpi=150, bbox_inches="tight")
plt.close()
print("     Saved → outputs/eda/02_monthly_orders.png")

# ─────────────────────────────────────────────────────────────
# CHART 3 — Payment Type Distribution
# ─────────────────────────────────────────────────────────────
print("[3/7] Drawing payment type chart...")
pay_counts = payments["payment_type"].value_counts()

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(pay_counts.values, labels=pay_counts.index,
       autopct="%1.1f%%", colors=COLORS,
       startangle=140, pctdistance=0.82,
       wedgeprops={"linewidth": 0.5, "edgecolor": "white"})
ax.set_title("Payment Method Distribution")
plt.tight_layout()
plt.savefig("outputs/eda/03_payment_types.png", dpi=150, bbox_inches="tight")
plt.close()
print("     Saved → outputs/eda/03_payment_types.png")

# ─────────────────────────────────────────────────────────────
# CHART 4 — Order Value Distribution
# ─────────────────────────────────────────────────────────────
print("[4/7] Drawing order value distribution...")
pay_per_order = payments.groupby("order_id")["payment_value"].sum().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(pay_per_order["payment_value"].clip(upper=1000),
             bins=50, color=COLORS[0], alpha=0.8, edgecolor="white")
axes[0].set_title("Order Value Distribution (up to R$1,000)")
axes[0].set_xlabel("Order Value (R$)")
axes[0].set_ylabel("Count")

axes[1].boxplot(pay_per_order["payment_value"].clip(upper=1000),
                vert=True, patch_artist=True,
                boxprops=dict(facecolor=COLORS[0], alpha=0.7),
                medianprops=dict(color="white", linewidth=2))
axes[1].set_title("Order Value Boxplot")
axes[1].set_ylabel("Order Value (R$)")

plt.suptitle("Order Value Analysis", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/eda/04_order_value.png", dpi=150, bbox_inches="tight")
plt.close()
print("     Saved → outputs/eda/04_order_value.png")

# ─────────────────────────────────────────────────────────────
# CHART 5 — Top 10 States by Customers
# ─────────────────────────────────────────────────────────────
print("[5/7] Drawing top states chart...")
top_states = customers["customer_state"].value_counts().head(10)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(top_states.index, top_states.values,
              color=COLORS[1], alpha=0.85)
for bar, val in zip(bars, top_states.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{val:,}", ha="center", fontsize=9)
ax.set_title("Top 10 States by Customer Count")
ax.set_xlabel("State")
ax.set_ylabel("Number of Customers")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig("outputs/eda/05_top_states.png", dpi=150, bbox_inches="tight")
plt.close()
print("     Saved → outputs/eda/05_top_states.png")

# ─────────────────────────────────────────────────────────────
# CHART 6 — Review Score Distribution
# ─────────────────────────────────────────────────────────────
print("[6/7] Drawing review score chart...")
review_counts = reviews["review_score"].value_counts().sort_index()
score_colors  = ["#E24B4A", "#EF9F27", "#EF9F27", "#1D9E75", "#1D9E75"]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(review_counts.index, review_counts.values,
              color=score_colors, edgecolor="white")
for bar, val in zip(bars, review_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 300,
            f"{val:,}", ha="center", fontsize=9)
ax.set_title("Customer Review Score Distribution")
ax.set_xlabel("Review Score  (1 = Worst  |  5 = Best)")
ax.set_ylabel("Number of Reviews")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig("outputs/eda/06_review_scores.png", dpi=150, bbox_inches="tight")
plt.close()
print("     Saved → outputs/eda/06_review_scores.png")

# ─────────────────────────────────────────────────────────────
# CHART 7 — Delivery Time Distribution
# ─────────────────────────────────────────────────────────────
print("[7/7] Drawing delivery time chart...")
orders_del = orders[orders["order_status"] == "delivered"].copy()
orders_del["delivery_days"] = (
    orders_del["order_delivered_customer_date"] -
    orders_del["order_purchase_timestamp"]
).dt.days
orders_del = orders_del.dropna(subset=["delivery_days"])
orders_del = orders_del[orders_del["delivery_days"].between(0, 60)]

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(orders_del["delivery_days"], bins=60,
        color=COLORS[2], alpha=0.8, edgecolor="white")
ax.axvline(orders_del["delivery_days"].median(),
           color=COLORS[3], linestyle="--", linewidth=2,
           label=f"Median: {orders_del['delivery_days'].median():.0f} days")
ax.set_title("Delivery Time Distribution")
ax.set_xlabel("Delivery Days")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/eda/07_delivery_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("     Saved → outputs/eda/07_delivery_time.png")

# ─────────────────────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EDA Summary")
print("=" * 60)
print(f"  Total orders        : {len(orders):,}")
print(f"  Delivered orders    : {len(delivered):,}")
print(f"  Unique customers    : {customers['customer_unique_id'].nunique():,}")
print(f"  Total revenue (R$)  : {payments['payment_value'].sum():,.2f}")
print(f"  Avg order value(R$) : {pay_per_order['payment_value'].mean():,.2f}")
print(f"  Avg review score    : {reviews['review_score'].mean():.2f} / 5.0")
print("=" * 60)
print("\n✅  EDA complete. All 7 charts saved to outputs/eda/")
print("    Next → run:  python step3_rfm_pyspark.py")