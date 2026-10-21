# pyspark_ml.py - PySpark for large-scale ML preprocessing
# Install: pip install pyspark
# Run: spark-submit pyspark_ml.py OR in Databricks/EMR notebooks

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType
from pyspark.ml import Pipeline as SparkPipeline
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler as SparkScaler,
    StringIndexer, OneHotEncoder, Bucketizer
)
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import numpy as np

# ── Initialize Spark session ──────────────────────────────────────
spark = SparkSession.builder \
    .appName('ML-Feature-Engineering') \
    .config('spark.executor.memory', '4g') \
    .config('spark.executor.cores', '4') \
    .config('spark.sql.shuffle.partitions', '200') \
    .getOrCreate()

spark.sparkContext.setLogLevel('WARN')
print(f'Spark version: {spark.version}')

# ── Generate sample large dataset ────────────────────────────────
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
pdf = pd.DataFrame(data.data, columns=data.feature_names)
pdf['target'] = data.target.astype(float)

# Replicate to simulate large dataset
pdf_large = pd.concat([pdf] * 100, ignore_index=True)  # ~57,000 rows

# Convert to Spark DataFrame
sdf = spark.createDataFrame(pdf_large)
print(f'Spark DataFrame: {sdf.count():,} rows × {len(sdf.columns)} columns')

# ── Spark SQL for feature engineering ────────────────────────────
sdf.createOrReplaceTempView('cancer_data')

sdf_engineered = spark.sql('''
    SELECT *,
        `mean area` / (`mean perimeter` + 0.0001) AS area_per_perimeter,
        `mean compactness` / (`mean smoothness` + 0.0001) AS compactness_ratio,
        LOG(`mean area` + 1) AS log_mean_area,
        CASE
            WHEN `mean radius` < 12 THEN 'small'
            WHEN `mean radius` < 18 THEN 'medium'
            ELSE 'large'
        END AS size_category
    FROM cancer_data
''')

print(f'After feature engineering: {len(sdf_engineered.columns)} columns')

# ── Spark DataFrame operations ────────────────────────────────────
# Aggregations
stats = sdf.groupBy('target').agg(
    F.count('*').alias('count'),
    F.mean('mean radius').alias('avg_radius'),
    F.stddev('mean radius').alias('std_radius'),
    F.percentile_approx('mean area', 0.5).alias('median_area')
)
print('\nClass statistics:')
stats.show()

# Window functions
from pyspark.sql.window import Window
window = Window.orderBy('mean radius')
sdf_with_rank = sdf.withColumn('radius_rank', F.rank().over(window))
sdf_with_rank = sdf_with_rank.withColumn('radius_percentile',
    F.percent_rank().over(window) * 100
)

# ── Spark MLlib Pipeline ─────────────────────────────────────────
# Prepare for MLlib: need to handle string column 'size_category'
sdf_ml = sdf_engineered.drop('size_category')  # Drop string for simplicity

# Feature column names
numeric_cols = [c for c in sdf_ml.columns if c != 'target']

# Assemble features into vector
assembler = VectorAssembler(
    inputCols=numeric_cols,
    outputCol='features_raw',
    handleInvalid='skip'
)

# Scale features
scaler = SparkScaler(
    inputCol='features_raw',
    outputCol='features',
    withStd=True, withMean=True
)

# Gradient Boosted Trees classifier
gbt = GBTClassifier(
    labelCol='target',
    featuresCol='features',
    maxIter=50,
    maxDepth=4,
    seed=42
)

# Build MLlib Pipeline
ml_pipeline = SparkPipeline(stages=[assembler, scaler, gbt])

# Train-test split
train_sdf, test_sdf = sdf_ml.randomSplit([0.8, 0.2], seed=42)
print(f'Train: {train_sdf.count():,}  Test: {test_sdf.count():,}')

# Fit pipeline
model_spark = ml_pipeline.fit(train_sdf)
predictions = model_spark.transform(test_sdf)

# Evaluate
evaluator = BinaryClassificationEvaluator(labelCol='target', metricName='areaUnderROC')
auc = evaluator.evaluate(predictions)
print(f'Spark GBT AUC: {auc:.4f}')

# ── Batch inference at scale ──────────────────────────────────────
print('\nBatch inference on full dataset:')
all_predictions = model_spark.transform(sdf_ml)
n_malignant = all_predictions.filter(F.col('prediction') == 0).count()
n_benign    = all_predictions.filter(F.col('prediction') == 1).count()
print(f'  Total: {all_predictions.count():,}')
print(f'  Malignant: {n_malignant:,}  Benign: {n_benign:,}')

spark.stop()
