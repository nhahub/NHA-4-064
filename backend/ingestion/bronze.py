from config import spark


raw_data = "/app/transactions-fraud-datasets/"
bronze_path = "/app/ingestion/bronze/"

# ==========================================
# 1. CSV file ingestion
# ==========================================
csv_tables = {
    'cards_data': 'cards_data.csv',
    'transactions_data': 'transactions_data.csv',
    'users_data': 'users_data.csv'
}

for name, file_name in csv_tables.items():
    print(f"🔃Start Ingesting CSV table: {name}")
    df_csv = (
        spark.read
        .option('header', True)
        .option('inferSchema', True)
        .csv(raw_data + file_name)
    )
    
    df_csv.write.mode('overwrite').parquet(f'{bronze_path}{name}')
    print(f"✅End Ingesting CSV table: {name}")


# ==========================================
# 2. JSON file ingestion
# ==========================================
json_tables = {
    'mcc_codes': 'mcc_codes.json',
    'train_fraud_labels': 'train_fraud_labels.json'
}

for name, file_name in json_tables.items():
    print(f"🔃Start Ingesting JSON table: {name}")
    df_json = (
        spark.read
        .option('multiLine', True) 
        .json(raw_data + file_name)
    )
    
    df_json.write.mode('overwrite').parquet(f'{bronze_path}{name}')
    print(f"✅End Ingesting JSON table: {name}")

print("Successfully ingested all tables into the Bronze layer!👏✅🔚")

spark.stop()