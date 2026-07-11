from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import LongType
import os

# ==========================================
# 🛠️ إعداد بيئة التشغيل للويندوز والجافا الحديثة
# ==========================================
os.environ['HADOOP_HOME'] = r"C:\hadoop"
os.environ['PATH'] += os.pathsep + r"C:\hadoop\bin"
os.environ['JDK_JAVA_OPTIONS'] = (
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
    "--add-opens=java.base/io.netty=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
    "--add-opens=java.base/java.net=ALL-UNNAMED "
    "--add-opens=java.base/java.text=ALL-UNNAMED "
    "--add-opens=java.base/java.nio=ALL-UNNAMED "
    "--add-opens=java.base/java.nio.channels=ALL-UNNAMED "
    "--add-opens=java.base/java.util.regex=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
    "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
    "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED "
    "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED"
)

SPARK_MASTER = os.environ.get('SPARK_MASTER', 'local[*]')

# 🚀 إنشاء الـ Spark Session مع تخصيص الذاكرة بناءً على رامات جهازك (16GB)
spark = (
    SparkSession.builder
    .appName('fraud_detection_gold')
    .master(SPARK_MASTER)
    .config("spark.driver.memory", "6g")          # تخصيص 6 جيجا للـ Driver لمنع الـ OutOfMemory
    .config("spark.executor.memory", "6g")        # تخصيص 6 جيجا للـ Executors
    .config("spark.sql.shuffle.partitions", "8")  # تقليل الأجزاء لتسريع الـ Shuffling على اللوكال
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    .config("spark.driver.extraJavaOptions", os.environ['JDK_JAVA_OPTIONS'])
    .config("spark.executor.extraJavaOptions", os.environ['JDK_JAVA_OPTIONS'])
    .getOrCreate()
)

# 🔥 تعطيل الـ ANSI Mode لمنع توقف الكود وتحويل القيم المشوهة لـ Null تلقائياً
spark.sql("SET spark.sql.ansi.enabled=false")

# ==========================================
# 📁 تحديد مسارات البيانات
# ==========================================
SILVER_PATH = r"C:\Users\ahmed sadiwy\Downloads\Fraud_Detection_Proj\Fraud_Detection_Proj\ingestion\silver"
GOLD_PATH = r"C:\Users\ahmed sadiwy\Downloads\Fraud_Detection_Proj\Fraud_Detection_Proj\ingestion\gold"

# 1. Load Silver Data 
transactions = spark.read.parquet(f"{SILVER_PATH}/transactions")
users = spark.read.parquet(f"{SILVER_PATH}/users")
cards = spark.read.parquet(f"{SILVER_PATH}/cards")
train_fraud_labels = spark.read.parquet(f"{SILVER_PATH}/fraud_labels")

# تحويل الأنواع إلى Long، وإذا كان هناك نص سيتحول إلى Null تلقائياً
transactions = transactions.withColumn("id", F.col("id").cast(LongType()))
train_fraud_labels = train_fraud_labels.withColumn("id", F.col("id").cast(LongType()))

# حذف الـ Nulls الموجودة في الـ ID
transactions = transactions.filter(F.col("id").isNotNull())
train_fraud_labels = train_fraud_labels.filter(F.col("id").isNotNull())

# دمج المعاملات مع عمود الـ Labels
transactions = transactions.join(train_fraud_labels, on="id", how="left")

# 2. Extract Time Features
transactions = transactions.withColumn("hour", F.hour("date")) \
                           .withColumn("day_of_week", F.dayofweek("date") - 1) \
                           .withColumn("month", F.month("date")) \
                           .withColumn("is_weekend", F.when(F.col("day_of_week").isin([5, 6]), 1).otherwise(0)) \
                           .withColumn("is_night", F.when((F.col("hour") >= 0) & (F.col("hour") <= 5), 1).otherwise(0)) \
                           .withColumn("date_only", F.to_date("date"))

# 3. Client Stats (Feature Engineering)
client_window = Window.partitionBy("client_id")
transactions = transactions.withColumn("client_mean_amount", F.mean("amount").over(client_window)) \
                           .withColumn("client_std_amount", F.stddev("amount").over(client_window)) \
                           .withColumn("client_max_amount", F.max("amount").over(client_window)) \
                           .withColumn("client_total_amount", F.sum("amount").over(client_window)) \
                           .withColumn("client_tx_count", F.count("id").over(client_window))

transactions = transactions.withColumn("amount_vs_client_mean", F.col("amount") / (F.col("client_mean_amount") + 1))

# 4. Join with Cards for Credit Limit
cards_features = cards.select(F.col("id").alias("card_id"), "client_id", "credit_limit")
transactions = transactions.join(cards_features, on=["card_id", "client_id"], how="left")
transactions = transactions.withColumn("amount_to_credit_ratio", F.col("amount") / (F.col("credit_limit") + 1))

# 5. Frequency Features
day_window = Window.partitionBy("client_id", "date_only")
transactions = transactions.withColumn("tx_count_same_day", F.count("id").over(day_window))

merchant_window = Window.partitionBy("client_id", "merchant_id")
transactions = transactions.withColumn("client_merchant_freq", F.count("id").over(merchant_window))

# 6. Boolean/Categorical Features
transactions = transactions.withColumn("is_online", F.when(F.col("use_chip") == "Online Transaction", 1).otherwise(0)) \
                           .withColumn("is_chip", F.when(F.col("use_chip") == "Chip Transaction", 1).otherwise(0)) \
                           .withColumn("is_swipe", F.when(F.col("use_chip") == "Swipe Transaction", 1).otherwise(0)) \
                           .withColumn("has_error", F.when(F.col("errors") != "No Erros", 1).otherwise(0))

# Target Column (تحويل آمن لعمود الـ Target)
target_col = "target" if "target" in transactions.columns else ("Target" if "Target" in transactions.columns else None)

if target_col:
    transactions = transactions.withColumn(
        "Target_Num", 
        F.when(F.lower(F.col(target_col)) == "yes", 1).otherwise(0)
    )

# 7. Select Final Features
final_features = [
    'id', 'client_id', 'card_id', 'amount', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_night',
    'client_mean_amount', 'client_std_amount', 'client_max_amount', 'client_tx_count', 'amount_vs_client_mean',
    'amount_to_credit_ratio', 'tx_count_same_day', 'client_merchant_freq', 'is_online', 'is_chip', 'is_swipe',
    'has_error'
]

if "Target_Num" in transactions.columns:
    final_features.append("Target_Num")

gold_df = transactions.select(final_features)

# ==========================================
# 🔍 8. فحص البيانات والـ Logging
# ==========================================
print("\n" + "="*40)
print("🔍 SYSTEM CHECK: DATA VERIFICATION")
print("="*40)
print(f"🔹 Total Rows in gold_df: {gold_df.count()}")
print("🔹 Sample Data:")
gold_df.show(5, truncate=False)

if "Target_Num" in gold_df.columns:
    print("🔹 Distribution of Target_Num:")
    gold_df.groupBy("Target_Num").count().show()
print("="*40 + "\n")

# 9. Save Gold Layer
gold_df.write.mode("overwrite").parquet(f"{GOLD_PATH}/master_table")

print(f"✅ Gold Layer master_table created and saved successfully to {GOLD_PATH}/master_table")

spark.stop()