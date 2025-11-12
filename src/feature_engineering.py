from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def create_feature_interactions(df):
    df = df.withColumn(
        "torque_per_rpm", 
        F.when(F.col("max_torque_rpm") > 0, 
               F.col("max_torque_nm") / F.col("max_torque_rpm"))
               .otherwise(None)
    )

    df = df.withColumn(
        "power_per_rpm",
        F.when(F.col("max_power_rpm") > 0, 
               F.col("max_power_bhp") / F.col("max_power_rpm"))
               .otherwise(None)
    )

    return df

if __name__ == "__main__":
    spark = SparkSession.builder.appName("InsuranceClaimsFeatureEngineering").getOrCreate()
    df = spark.read.parquet("data/preprocessed/insurance_claims_clean.parquet")
    df = create_feature_interactions(df)
    df.write.mode("overwrite").parquet("data/processed/insurance_claims_features.parquet")

    print("Feature engineering complete. Saved to data/processed/insurance_claims_features.parquet")

    spark.stop()