import os
from pyspark.sql import SparkSession

SPARK_MASTER = os.environ.get('SPARK_MASTER', 'local[*]')

spark = (
    SparkSession.builder
    .appName('fraud_detection')
    .master(SPARK_MASTER)
    .getOrCreate()
)