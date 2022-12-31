import os
import time

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.functions import col, when
from pyspark.sql.types import *

SCHEMA = StructType([StructField("Arrival_Time", LongType(), True),
                     StructField("Creation_Time", LongType(), True),
                     StructField("Device", StringType(), True),
                     StructField("Index", LongType(), True),
                     StructField("Model", StringType(), True),
                     StructField("User", StringType(), True),
                     StructField("gt", StringType(), True),
                     StructField("x", DoubleType(), True),
                     StructField("y", DoubleType(), True),
                     StructField("z", DoubleType(), True)])

spark = SparkSession.builder.appName('demo_app') \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .getOrCreate()

os.environ['PYSPARK_SUBMIT_ARGS'] = \
    "--packages=org.apache.spark:spark-sql-kafka-0-10_2.12:2.4.8,com.microsoft.azure:spark-mssql-connector:1.0.1"
kafka_server = 'dds2020s-kafka.eastus.cloudapp.azure.com:9092'
topic = "activities"

streaming = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_server) \
    .option("subscribe", topic) \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", False) \
    .option("maxOffsetsPerTrigger", 750000) \
    .load() \
    .select(f.from_json(f.decode("value", "US-ASCII"), schema=SCHEMA).alias("value")).select("value.*")


def orgenaize_data(data):
    data = data.filter(data.User != 'null')

    data = data.withColumn("User_id", when(col("User") == 'a', 1).when(col("User") == 'b', 2)
                           .when(col("User") == 'c', 3).when(col("User") == 'd', 4).
                           when(col("User") == 'e', 5).when(col("User") == 'f', 6) \
                           .when(col("User") == 'g', 7).when(col("User") == 'h', 8) \
                           .when(col("User") == 'i', 9))

    colm = static_df.columns
    assembler = VectorAssembler(
        inputCols=["User_id", "x", "y", "z", "Index", "Creation_Time"],
        outputCol="features")
    output = assembler.transform(data)
    output = output.filter(output.gt != 'null')
    output = output.withColumn("label", when(col("gt") == 'stand', 1).when(col("gt") == 'sit', 2)
                               .when(col("gt") == 'walk', 3).when(col("gt") == 'stairsup', 4).
                               when(col("gt") == 'stairsdown', 5).when(col("gt") == 'bike', 6))
    output = output.drop(*colm)
    return output


static_df = spark.read \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_server) \
    .option("subscribe", topic) \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", False) \
    .option("maxOffsetsPerTrigger", 10000) \
    .load() \
    .select(f.from_json(f.decode("value", "US-ASCII"), schema=SCHEMA).alias("value")).select("value.*")

train = static_df.sample(False, 0.05)
train = orgenaize_data(train)
train_row = train.count()
avg_accuracy = 0
rfClassifier = RandomForestClassifier(labelCol="label", featuresCol="features", maxBins=42, maxDepth=10, numTrees=50)


def foreach_batch_function(data, batch_id):
    global train, train_row, avg_accuracy

    time.sleep(10)
    print("in forach batch func")
    test_rows = data.count()
    test_data = orgenaize_data(data)
    rfClassifier = RandomForestClassifier(labelCol="label", featuresCol="features", maxBins=42, maxDepth=10,
                                          numTrees=50)
    pModel = rfClassifier.fit(train)
    cvPredDF = pModel.transform(test_data)
    mcEvaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    accuracy = mcEvaluator.evaluate(cvPredDF)
    print("\n\ttrain size: {}\t for batch: {}".format(train_row, batch_id))
    print("\ttest size: {}\t\t for batch: {}\n".format(test_rows, batch_id))
    print("\taccuracy: {}".format(accuracy))
    train = train.union(test_data)
    train_row += test_rows
    avg_accuracy = avg_accuracy + accuracy
    if batch_id>= 8:
        print("\taverge accuracy: {}".format(avg_accuracy/(batch_id+1)))
        print("Finishing streaming")




def main():
    global train_row, avg_accuracy


    simpleTransform = streaming \
        .select("gt", "model", "User", "x", "y", "z", "Creation_Time", "Index") \
        .writeStream \
        .foreachBatch(foreach_batch_function) \
        .start() \
        .awaitTermination()

if __name__ == "__main__":
    main()


