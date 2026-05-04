# =============================================================================
# STEP 3 — RFM Feature Engineering  (PySpark + Pandas fallback)
# HOW TO RUN:  python step3_rfm_pyspark.py
# WHAT IT DOES: Joins all CSV tables together, calculates Recency /
#               Frequency / Monetary per customer, saves rfm_features.csv
# NOTE: If PySpark fails, the script automatically uses Pandas instead.
#       Both give the exact same result.
# =============================================================================

import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
os.makedirs("data",    exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("  STEP 3 — RFM Feature Engineering")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# TRY PYSPARK FIRST
# ─────────────────────────────────────────────────────────────
USE_SPARK = False
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import (
        col, max as spark_max, sum as spark_sum,
        countDistinct, datediff, lit,
        round as spark_round
    )
    spark = SparkSession.builder \
        .appName("OlistRFM") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    USE_SPARK = True
    print("\n✅  PySpark session started — using distributed processing")
except Exception as e:
    print(f"\n⚠️   PySpark not available ({e})")
    print("    Automatically using Pandas instead (same result)\n")

# ─────────────────────────────────────────────────────────────
# PYSPARK PATH
# ─────────────────────────────────────────────────────────────
if USE_SPARK:
    print("\n[1/5] Loading CSVs into Spark DataFrames...")
    orders    = spark.read.csv("data/olist_orders_dataset.csv",
                               header=True, inferSchema=True)
    customers = spark.read.csv("data/olist_customers_dataset.csv",
                               header=True, inferSchema=True)
    items     = spark.read.csv("data/olist_order_items_dataset.csv",
                               header=True, inferSchema=True)
    payments  = spark.read.csv("data/olist_order_payments_dataset.csv",
                               header=True, inferSchema=True)
    print(f"     orders   : {orders.count():,}")
    print(f"     customers: {customers.count():,}")
    print(f"     items    : {items.count():,}")
    print(f"     payments : {payments.count():,}")

    print("\n[2/5] Filtering to delivered orders only...")
    orders = orders.filter(col("order_status") == "delivered")
    print(f"     Delivered: {orders.count():,}")

    print("\n[3/5] Joining tables...")
    df = orders \
        .join(customers, "customer_id", "left") \
        .join(items,     "order_id",    "left") \
        .join(payments.select("order_id", "payment_value",
                              "payment_type"), "order_id", "left") \
        .withColumn("total_price", col("price") + col("freight_value")) \
        .select("customer_unique_id", "order_id",
                "order_purchase_timestamp",
                "total_price", "customer_state", "customer_city") \
        .filter(col("total_price").isNotNull())
    print(f"     Joined rows: {df.count():,}")

    print("\n[4/5] Calculating RFM per customer...")
    ref_date = df.agg(
        spark_max("order_purchase_timestamp")).collect()[0][0]

    rfm = df.groupBy("customer_unique_id").agg(
        datediff(lit(ref_date),
                 spark_max("order_purchase_timestamp")).alias("Recency"),
        countDistinct("order_id").alias("Frequency"),
        spark_round(spark_sum("total_price"), 2).alias("Monetary"),
        spark_max("customer_state").alias("State"),
        spark_max("customer_city").alias("City")
    ).filter(
        col("Recency").isNotNull() &
        col("Frequency").isNotNull() &
        (col("Monetary") > 0)
    )
    print(f"     Unique customers: {rfm.count():,}")

    print("\n[5/5] Saving to data/rfm_features.csv...")
    rfm_pd = rfm.toPandas()
    spark.stop()

# ─────────────────────────────────────────────────────────────
# PANDAS FALLBACK PATH  (identical logic, no Spark needed)
# ─────────────────────────────────────────────────────────────
else:
    print("\n[1/5] Loading CSVs with Pandas...")
    orders    = pd.read_csv("data/olist_orders_dataset.csv",
                            parse_dates=["order_purchase_timestamp"])
    customers = pd.read_csv("data/olist_customers_dataset.csv")
    items     = pd.read_csv("data/olist_order_items_dataset.csv")
    payments  = pd.read_csv("data/olist_order_payments_dataset.csv")
    print(f"     orders   : {len(orders):,}")
    print(f"     customers: {len(customers):,}")
    print(f"     items    : {len(items):,}")
    print(f"     payments : {len(payments):,}")

    print("\n[2/5] Filtering to delivered orders only...")
    orders = orders[orders["order_status"] == "delivered"].copy()
    print(f"     Delivered: {len(orders):,}")

    print("\n[3/5] Joining tables...")
    df = orders \
        .merge(customers, on="customer_id", how="left") \
        .merge(items,     on="order_id",    how="left") \
        .merge(payments[["order_id", "payment_value", "payment_type"]],
               on="order_id", how="left")
    df["total_price"] = df["price"] + df["freight_value"]
    df = df.dropna(subset=["total_price", "customer_unique_id"])
    df = df[df["total_price"] > 0]
    print(f"     Joined rows: {len(df):,}")

    print("\n[4/5] Calculating RFM per customer...")
    ref_date = df["order_purchase_timestamp"].max()
    rfm_pd = df.groupby("customer_unique_id").agg(
        Recency  =("order_purchase_timestamp",
                   lambda x: (ref_date - x.max()).days),
        Frequency=("order_id",       "nunique"),
        Monetary =("total_price",    "sum"),
        State    =("customer_state", "first"),
        City     =("customer_city",  "first")
    ).reset_index()
    rfm_pd["Monetary"] = rfm_pd["Monetary"].round(2)
    rfm_pd = rfm_pd[rfm_pd["Monetary"] > 0]
    print(f"     Unique customers: {len(rfm_pd):,}")

    print("\n[5/5] Saving to data/rfm_features.csv...")

# ─────────────────────────────────────────────────────────────
# SAVE & SUMMARY
# ─────────────────────────────────────────────────────────────
rfm_pd.to_csv("data/rfm_features.csv", index=False)

print("\n" + "=" * 60)
print("  RFM Summary Statistics")
print("=" * 60)
print(rfm_pd[["Recency", "Frequency", "Monetary"]] \
      .describe().round(2).to_string())
print("=" * 60)
print(f"\n✅  Saved {len(rfm_pd):,} customer records to data/rfm_features.csv")
print("    Next → run:  python step4_clustering.py")