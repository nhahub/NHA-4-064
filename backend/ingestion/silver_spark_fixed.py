from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

import os
import json
from pathlib import Path

SPARK_MASTER = os.environ.get('SPARK_MASTER', 'local[*]')

spark = (
    SparkSession.builder
    .appName('fraud_detection')
    .master(SPARK_MASTER)
    .getOrCreate()
)

BASE_PATH = "/app/ingestion/bronze"
SILVER_PATH = "/app/ingestion/silver"

transactions_raw = spark.read.parquet(f"{BASE_PATH}/transactions_data")
cards_raw        = spark.read.parquet(f"{BASE_PATH}/cards_data")
users_raw        = spark.read.parquet(f"{BASE_PATH}/users_data")
mcc_raw          = spark.read.parquet(f"{BASE_PATH}/mcc_codes")

labels_ndjson_path = Path(f"{BASE_PATH}/train_fraud_labels.ndjson")
labels_json_path = Path("/app/transactions-fraud-datasets/train_fraud_labels.json")


def convert_labels_json_to_ndjson_safe(json_path: Path, ndjson_path: Path) -> None:
    # streaming character-level parser to avoid loading entire JSON in memory
    if ndjson_path.exists():
        try:
            ndjson_path.unlink()
        except Exception:
            pass
    print(f"Converting labels JSON to NDJSON (safe): {json_path} -> {ndjson_path}")
    ndjson_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("r", encoding="utf-8") as src, ndjson_path.open("w", encoding="utf-8") as dst:
        write = dst.write
        state = "search_key"
        key = None
        val = None
        buffer = []
        while True:
            ch = src.read(1)
            if not ch:
                break
            if state == "search_key":
                if ch == '"':
                    buffer = []
                    state = "reading_key"
            elif state == "reading_key":
                if ch == '\\':
                    next_ch = src.read(1)
                    if not next_ch:
                        break
                    buffer.append(next_ch)
                elif ch == '"':
                    key = ''.join(buffer)
                    state = "search_value"
                else:
                    buffer.append(ch)
            elif state == "search_value":
                if ch == ':':
                    # consume until opening quote of value
                    while True:
                        ch2 = src.read(1)
                        if not ch2:
                            break
                        if ch2 == '"':
                            buffer = []
                            state = "reading_value"
                            break
                # else keep scanning
            elif state == "reading_value":
                if ch == '\\':
                    next_ch = src.read(1)
                    if not next_ch:
                        break
                    buffer.append(next_ch)
                elif ch == '"':
                    val = ''.join(buffer)
                    # write ndjson line
                    write(json.dumps({"id": key, "target": val}) + "\n")
                    state = "search_key"
                else:
                    buffer.append(ch)


# Ensure labels NDJSON exists
if labels_ndjson_path.exists():
    labels_raw = spark.read.json(str(labels_ndjson_path))
else:
    convert_labels_json_to_ndjson_safe(labels_json_path, labels_ndjson_path)
    labels_raw = spark.read.json(str(labels_ndjson_path))

print("Transactions:", transactions_raw.count(), transactions_raw.columns)
print("Cards:       ", cards_raw.count(),        cards_raw.columns)
print("Users:       ", users_raw.count(),         users_raw.columns)

# Transactions cleaning
transactions = transactions_raw \
    .withColumn("date",   F.to_timestamp("date")) \
    .withColumn("amount", F.regexp_replace("amount", r"\$", "").cast(DoubleType()))

transactions = transactions.withColumn(
    "merchant_state",
    F.when(
        F.col("merchant_state").isNull() & (F.col("merchant_city") == "ONLINE"),
        F.lit("ONLINE")
    ).otherwise(F.col("merchant_state"))
)

transactions = transactions.withColumn(
    "errors",
    F.coalesce(F.col("errors"), F.lit("No Errors"))
)

# keep zip mapping small for brevity — reuse original mapping if needed
zip_map_expr = F.when(F.col("merchant_city") == "ONLINE", F.lit("0")).otherwise(F.col("zip"))
transactions = transactions.withColumn(
    "zip",
    F.when(F.col("zip").isNull(), zip_map_expr).otherwise(F.col("zip"))
)
transactions = transactions.withColumn(
    "zip",
    F.coalesce(F.col("zip"), F.lit("01000-000"))
)

# Users cleaning
users = users_raw \
    .withColumn("per_capita_income", F.regexp_replace("per_capita_income", r"\$", "").cast(DoubleType())) \
    .withColumn("yearly_income",     F.regexp_replace("yearly_income",     r"\$", "").cast(DoubleType())) \
    .withColumn("total_debt",        F.regexp_replace("total_debt",        r"\$", "").cast(DoubleType()))

users = users.withColumn(
    "time_left_until_retirement",
    F.when(
        F.col("current_age") >= F.col("retirement_age"),
        F.lit("retired")
    ).otherwise(
        (F.col("retirement_age") - F.col("current_age")).cast("string")
    )
)

# Cards cleaning
cards = cards_raw \
    .withColumn("credit_limit",  F.regexp_replace("credit_limit", r"\$", "").cast(DoubleType())) \
    .withColumn("expires",       F.to_date("expires",       "MM/yyyy")) \
    .withColumn("acct_open_date",F.to_date("acct_open_date","MM/yyyy"))

cards = cards.withColumn(
    "account_duration_days",
    F.datediff(F.col("expires"), F.col("acct_open_date"))
)

# MCC and labels
mcc_codes = mcc_raw
train_fraud_labels = labels_raw

# Joins
transactions_labels = transactions.join(train_fraud_labels, on="id", how="left")

# Save Silver layer to Parquet
transactions.write.mode("overwrite").parquet(f"{SILVER_PATH}/transactions")
users.write.mode("overwrite").parquet(f"{SILVER_PATH}/users")
cards.write.mode("overwrite").parquet(f"{SILVER_PATH}/cards")
mcc_codes.write.mode("overwrite").parquet(f"{SILVER_PATH}/mcc_codes")
train_fraud_labels.write.mode("overwrite").parquet(f"{SILVER_PATH}/fraud_labels")
transactions_labels.write.mode("overwrite").parquet(f"{SILVER_PATH}/transactions_with_labels")

print("✅ Silver Layer saved successfully to", SILVER_PATH)

spark.stop()
