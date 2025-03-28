from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(X, y, use_spark=False):
    """
    Train a Random Forest classifier using scikit-learn or Spark MLlib based on the use_spark flag.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    if use_spark:
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.classification import RandomForestClassifier as SparkRF
        feature_cols = list(X.columns)
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        rf = SparkRF(labelCol="label", featuresCol="features", numTrees=100)
        pipeline = Pipeline(stages=[assembler, rf])
        train_df = X_train.copy()
        train_df['label'] = y_train
        test_df = X_test.copy()
        test_df['label'] = y_test
        spark = __import__("pyspark.sql").SparkSession.builder.appName("ReadmissionProject").getOrCreate()
        train_spark = spark.createDataFrame(train_df)
        test_spark = spark.createDataFrame(test_df)
        model = pipeline.fit(train_spark)
        return model, X_test, y_test
    else:
        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        clf.fit(X_train, y_train)
        model = clf
        return model, X_test, y_test
