import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.stat import Correlation

DATA_PATH = "Power_Plant.csv"
SEED = 42
TOP_SOURCES = ["SUN", "NG", "WAT", "BAT", "WND", "LFG", "GEO"]
NUMERIC = ["StartYear", "x", "y", "CEC_Jurisdictional"]

spark.sparkContext.setLogLevel("ERROR")
spark.conf.set("spark.sql.ansi.enabled", "false")


def save(name):
    plt.tight_layout()
    plt.savefig(name, dpi=130)
    plt.close()
    print("Saved", name)


def load_data():
    raw = (
        spark.read.csv(DATA_PATH, header=True, inferSchema=True)
        .withColumnRenamed("Retired Plant", "Retired")
        .withColumnRenamed("Capacity_Latest", "Capacity")
        .withColumnRenamed("PriEnergySource", "Source")
    )
    print("Total records:", raw.count())

    df = raw.filter(
        (F.col("Retired") == 0) & (F.col("Capacity") > 0)
        & F.col("Source").isNotNull() & F.col("StartDate").isNotNull()
    )

    for c in ["x", "y", "CEC_Jurisdictional", "Capacity"]:
        df = df.withColumn(c, F.col(c).cast("double"))

    df = (
        df.withColumn("Source2", F.when(F.col("Source").isin(TOP_SOURCES), F.col("Source")).otherwise("OTHER"))
          .withColumn("StartYear", F.year(F.to_timestamp("StartDate", "M/d/yyyy h:mm:ss a")).cast("double"))
          .withColumn("LogCap", F.log1p("Capacity"))
          .select("PlantName", "Source2", *NUMERIC, "Capacity", "LogCap")
          .dropna()
    )
    print("Usable records:", df.count())
    df.show(5)
    return df


def plot_correlation(df):
    cols = ["LogCap"] + NUMERIC
    vec = VectorAssembler(inputCols=cols, outputCol="vec").transform(df)
    corr = Correlation.corr(vec, "vec", "pearson").first()[0].toArray()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                xticklabels=cols, yticklabels=cols)
    plt.title("Correlation Heatmap")
    save("lr_correlation_heatmap.png")


def build_pipeline():
    return Pipeline(stages=[
        StringIndexer(inputCol="Source2", outputCol="source_idx", handleInvalid="keep"),
        OneHotEncoder(inputCols=["source_idx"], outputCols=["source_vec"]),
        VectorAssembler(inputCols=NUMERIC + ["source_vec"], outputCol="raw_features"),
        StandardScaler(inputCol="raw_features", outputCol="features"),
        LinearRegression(featuresCol="features", labelCol="LogCap"),
    ])


def evaluate(model, train, preds):
    evaluator = RegressionEvaluator(labelCol="LogCap", predictionCol="prediction")

    for metric in ["mse", "rmse", "r2", "mae"]:
        score = evaluator.setMetricName(metric).evaluate(preds)
        print(f"{metric.upper():<5}: {score:.4f}")

    train_r2 = evaluator.setMetricName("r2").evaluate(model.transform(train))
    print(f"Train R2: {train_r2:.4f}")


def analyze(preds):
    preds = preds.withColumn("err", F.col("LogCap") - F.col("prediction")) \
                 .withColumn("abs_err", F.abs("err"))

    print("Error by energy source:")
    (preds.groupBy("Source2")
          .agg(F.count("*").alias("plants"),
               F.round(F.avg(F.pow("err", 2)), 3).alias("mse"),
               F.round(F.sqrt(F.avg(F.pow("err", 2))), 3).alias("rmse"),
               F.round(F.avg("abs_err"), 3).alias("mae"))
          .orderBy(F.desc("rmse"))
          .show())

    print("5 biggest misses (in MW):")
    (preds.select("PlantName", "Source2",
                  F.round("Capacity", 1).alias("actual_MW"),
                  F.round(F.expm1("prediction"), 1).alias("predicted_MW"),
                  F.round(F.abs(F.col("Capacity") - F.expm1("prediction")), 1).alias("error_MW"))
          .orderBy(F.desc("error_MW"))
          .show(5, truncate=40))

    return preds


def plot_results(df, preds):
    by_decade = (
        preds.filter(F.col("StartYear") >= 1980)
        .withColumn("decade", (F.floor(F.col("StartYear") / 10) * 10).cast("int"))
        .groupBy("Source2", "decade").agg(F.avg("abs_err").alias("mae"))
        .toPandas()
        .pivot(index="Source2", columns="decade", values="mae")
    )
    if by_decade.empty:
        print("No data for error heatmap.")
    else:
        by_decade.columns = [f"{d}s" for d in by_decade.columns]
        plt.figure(figsize=(8, 5))
        sns.heatmap(by_decade, annot=True, fmt=".2f", cmap="YlOrRd")
        plt.title("Prediction Error by Energy Source and Decade")
        plt.xlabel("Plant Start Decade")
        plt.ylabel("Energy Source")
        save("lr_error_heatmap.png")

    capacity = (df.groupBy("Source2").agg(F.sum("Capacity").alias("mw"))
                  .orderBy(F.desc("mw")).toPandas())
    rmse = (preds.groupBy("Source2").agg(F.sqrt(F.avg(F.pow("err", 2))).alias("rmse"))
                 .orderBy(F.desc("rmse")).toPandas())

    _, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].bar(capacity["Source2"], capacity["mw"])
    ax[0].set(title="Installed Capacity by Energy Source", xlabel="Energy Source", ylabel="Total Capacity (MW)")
    ax[1].bar(rmse["Source2"], rmse["rmse"])
    ax[1].set(title="RMSE by Energy Source", xlabel="Energy Source", ylabel="RMSE")
    save("lr_bar_graphs.png")

    year = F.col("StartYear").cast("int").alias("year")
    yearly = (df.filter(F.col("StartYear") >= 1990).groupBy(year)
                .agg(F.sum("Capacity").alias("mw")).orderBy("year").toPandas())
    trend = (preds.filter(F.col("StartYear") >= 1990).groupBy(year)
                  .agg(F.avg("LogCap").alias("actual"), F.avg("prediction").alias("predicted"))
                  .orderBy("year").toPandas())

    _, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(yearly["year"], yearly["mw"], marker="o")
    ax[0].set(title="Annual Installed Capacity", xlabel="Year", ylabel="Capacity (MW)")
    ax[1].plot(trend["year"], trend["actual"], marker="o", label="Actual")
    ax[1].plot(trend["year"], trend["predicted"], marker="s", linestyle="--", label="Predicted")
    ax[1].set(title="Actual vs Predicted Capacity", xlabel="Year", ylabel="Log Capacity")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    save("lr_line_graphs.png")


def main():
    df = load_data()
    plot_correlation(df)

    train, test = df.randomSplit([0.8, 0.2], seed=SEED)
    print("Train / test records:", train.count(), "/", test.count())

    model = build_pipeline().fit(train)
    preds = model.transform(test)

    lr = model.stages[-1]
    print("Intercept:", lr.intercept)
    print("Coefficients:", len(lr.coefficients))

    evaluate(model, train, preds)
    preds = analyze(preds)
    plot_results(df, preds)

    print("Linear Regression pipeline finished.")


if __name__ == "__main__":
    main()