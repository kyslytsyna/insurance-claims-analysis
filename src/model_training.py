from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql.window import Window



def compute_class_weights(df, label_col="claim_status"):
    pos_count = df.filter(F.col(label_col) == 1).count()
    neg_count = df.filter(F.col(label_col) == 0).count()
    total_count = pos_count + neg_count

    weight_for_0 = total_count / (2.0 * neg_count)
    weight_for_1 = total_count / (2.0 * pos_count)

    df_weighted = df.withColumn(
        "weight",
        when(F.col(label_col) == 0, weight_for_0).otherwise(weight_for_1)
    )
    return df_weighted



def load_and_prepare_data(spark, path:str):
    # Load parquet with engineered features
    df = spark.read.parquet(path)

    # Define numeric, categorical and binary features
    numeric_features = [
        "subscription_length",
        "vehicle_age",
        "customer_age",
        "region_density",
        "airbags",
        "displacement",
        "cylinder",
        "turning_radius",
        "length",
        "width",
        "gross_weight",
        "max_torque_nm",
        "max_torque_rpm",
        "max_power_bhp",
        "max_power_rpm",
        "torque_per_rpm",
        "power_per_rpm",
        "ncap_rating"
    ]

    categorical_features = [
        "region_code",
        "segment",
        "model",
        "fuel_type",
        "engine_type",
        "rear_brakes_type",
        "transmission_type",
        "steering_type"
    ]

    binary_features = [c for c in df.columns if c.startswith("is_")]

    return df, numeric_features, binary_features, categorical_features

def train_test_split(df, test_size=0.2, seed=42):
    train_df, test_df = df.randomSplit([1 - test_size, test_size], seed=seed)
    return train_df, test_df



# Logistic Regression
# ----------------------
def build_logistic_regression_model(numeric_features, binary_features, categorical_features):
    indexer = StringIndexer(
        inputCols=categorical_features,
        outputCols=[f"{c}_idx" for c in categorical_features],
        handleInvalid="keep")
    
    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in categorical_features],
        outputCols=[f"{c}_ohe" for c in categorical_features]
    )

    assembler = VectorAssembler(
        inputCols=numeric_features + binary_features + [f"{c}_ohe" for c in categorical_features],
        outputCol="features_unscaled"
    )

    scaler = StandardScaler(
        inputCol="features_unscaled",
        outputCol="features",
        withMean=True,
        withStd=True
    )

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="claim_status",
        weightCol="weight"
    )

    pipeline = Pipeline(stages=[indexer, encoder, assembler, scaler, lr])
    return pipeline



# Random Forest
# ----------------
def build_random_forest_model(numeric_features, binary_features, categorical_features):
    indexer = StringIndexer(
        inputCols=categorical_features,
        outputCols=[f"{c}_idx" for c in categorical_features],
        handleInvalid="keep")
    
    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in categorical_features],
        outputCols=[f"{c}_ohe" for c in categorical_features]
    )

    assembler = VectorAssembler(
        inputCols=numeric_features + binary_features + [f"{c}_ohe" for c in categorical_features],
        outputCol="features"
    )

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="claim_status"
    )

    pipeline = Pipeline(stages=[indexer, encoder ,assembler, rf])
    return pipeline



# Gradient Boosted Trees
# -------------------------
def build_gbt_model(numeric_features, binary_features, categorical_features):
    indexer = StringIndexer(
        inputCols=categorical_features,
        outputCols=[f"{c}_idx" for c in categorical_features],
        handleInvalid="keep")
    
    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in categorical_features],
        outputCols=[f"{c}_ohe" for c in categorical_features]
    )

    assembler = VectorAssembler(
        inputCols=numeric_features + binary_features + [f"{c}_ohe" for c in categorical_features],
        outputCol="features"
    )

    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="claim_status"
    )

    pipeline = Pipeline(stages=[indexer, encoder, assembler, gbt])
    return pipeline



# Evaluation Functions
# -----------------------
def evaluate(predictions, label_col="claim_status"):
    evaluator_auc = BinaryClassificationEvaluator(
        labelCol=label_col,
        metricName="areaUnderROC"
    )
    evaluator_pr = BinaryClassificationEvaluator(
        labelCol=label_col,
        metricName="areaUnderPR"
    )
    auc = evaluator_auc.evaluate(predictions)
    pr_auc = evaluator_pr.evaluate(predictions)
    print(f"AUC: {auc:.4f}, PR AUC: {pr_auc:.4f}")

    # Calibration
    prob_df = predictions.select(
        label_col,
        vector_to_array("probability")[1].alias("p")
    )

    bucketed = prob_df.withColumn(
        "bucket", (F.col("p") * 10).cast("int")
    )

    calib = bucketed.groupBy("bucket").agg(
        F.avg("p").alias("avg_pred"),
        F.avg(label_col).alias("avg_actual"),
        F.count("*").alias("count")
    ).orderBy("bucket")

    print("\nCalibration table (avg_pred vs avg_actual):")
    calib.show(20, truncate=False)

    # Lift Calculation
    window = Window.orderBy(F.col("p").desc())
    ranked = prob_df.withColumn("rank", F.row_number().over(window))

    total = ranked.count()
    top10 = ranked.filter(F.col("rank") <= total * 0.1)

    overall_rate = prob_df.agg(F.avg(label_col)).first()[0]
    top10_rate = top10.agg(F.avg(label_col)).first()[0]

    lift = top10_rate / overall_rate if overall_rate > 0 else 0

    print(f"\nLift (Top 10% risk customers): {lift:.2f}")



# Main block
# -------------
if __name__ == "__main__":
    spark = SparkSession.builder.appName("InsuranceClaimsModelTraining").getOrCreate()
    # Load and prepare data
    df, numeric_features, binary_features, categorical_features = load_and_prepare_data(
        spark, "data/processed/insurance_claims_features.parquet"
    )
    # Compute class weights
    df = compute_class_weights(df)

    # Train-test split
    train_df, test_df = train_test_split(df, test_size=0.2, seed=42)

    # Train & evaluate Logistic Regression
    print("Training Logistic Regression...")
    lr_pipeline = build_logistic_regression_model(numeric_features, binary_features, categorical_features)
    lr_model = lr_pipeline.fit(train_df)
    lr_predictions = lr_model.transform(test_df)
    print("Logistic Regression Evaluation:")
    evaluate(lr_predictions)

    # Train & evaluate Random Forest
    print("Training Random Forest...")
    rf_pipeline = build_random_forest_model(numeric_features, binary_features, categorical_features)
    rf_model = rf_pipeline.fit(train_df)
    rf_predictions = rf_model.transform(test_df)
    print("Random Forest Evaluation:")
    evaluate(rf_predictions)

    # Train & evaluate Gradient Boosted Trees
    print("Training Gradient Boosted Trees...")
    gbt_pipeline = build_gbt_model(numeric_features, binary_features, categorical_features)
    gbt_model = gbt_pipeline.fit(train_df)
    gbt_predictions = gbt_model.transform(test_df)
    print("Gradient Boosted Trees Evaluation:")
    evaluate(gbt_predictions)

    spark.stop()

