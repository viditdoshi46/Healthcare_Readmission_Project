import pandas as pd
try:
    from pyspark.sql import SparkSession
except ImportError:
    SparkSession = None

def load_data(file_path, use_spark=False):
    """
    Load dataset using pandas or PySpark based on the use_spark flag.
    """
    if use_spark and SparkSession is not None:
        spark = SparkSession.builder.appName("ReadmissionProject").getOrCreate()
        df = spark.read.csv(file_path, header=True, inferSchema=True)
    else:
        df = pd.read_csv(file_path)
    return df
