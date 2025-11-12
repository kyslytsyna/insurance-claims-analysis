from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import col, when, regexp_extract, trim


# Outlier removal skipped for now — distributions appear normal (see 01_EDA.ipynb)
# revisit after model evaluation if needed
def remove_outliers_iqr(df, column):
    quantiles = df.approxQuantile(column, [0.25, 0.75], 0.05)
    Q1 = quantiles[0]
    Q3 = quantiles[1]
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df.filter((col(column) >= lower_bound) & (col(column) <= upper_bound))
    return df_filtered

def load_and_clean_data(spark, path:str):
    df = spark.read.csv(path, header=True, inferSchema=True)

    # Drop unnecessary ID column
    df = df.drop("policy_id")

    # Encode Yes\No columns
    yes_no_cols = [c for c in df.columns if c.startswith("is_")]
    for c in yes_no_cols:
        df = df.withColumn(c, when(col(c) == "Yes", 1).when(col(c) == "No", 0).otherwise(None))

    # Extract numeric values from mixed columns
    df = df.withColumn(
        "max_torque_nm", 
        regexp_extract(trim(F.lower(col("max_torque"))), r"(\d+\.?\d*)", 1).cast(T.DoubleType())
        )
    df = df.withColumn(
        "max_torque_rpm",
        regexp_extract(trim(F.lower(col("max_torque"))), r"@(\d+)", 1).cast(T.DoubleType())
        ).drop("max_torque")
    df = df.withColumn(
        "max_power_bhp",
        regexp_extract(trim(F.lower(col("max_power"))), r"(\d+\.?\d*)", 1).cast(T.DoubleType())
        )
    df = df.withColumn(
        "max_power_rpm",
        regexp_extract(trim(F.lower(col("max_power"))), r"@(\d+)", 1).cast(T.DoubleType())
        ).drop("max_power")
    
    # Outlier removal can be added here if needed
    # numeric_cols = [
    # "subscription_length",
    # "vehicle_age",
    # "customer_age",
    # "region_density",
    # "airbags",
    # "displacement",
    # "cylinder",
    # "turning_radius",
    # "length",
    # "width",
    # "gross_weight",
    # "max_torque_nm",
    # "max_torque_rpm",
    # "max_power_bhp",
    # "max_power_rpm"
    # ]
    # for col_name in numeric_cols:
    #     df = remove_outliers_iqr(df, col_name)

    df.printSchema()

    return df

if __name__ == "__main__":
    spark = SparkSession.builder.appName("InsuranceClaimsPreprocessing").getOrCreate()
    df = load_and_clean_data(spark, "data/raw/insurance_claims_data.csv")
    df.write.mode("overwrite").parquet("data/preprocessed/insurance_claims_clean.parquet")

    print(" Data cleaning complete. Saved to data/preprocessed/insurance_claims_clean.parquet")

    spark.stop()