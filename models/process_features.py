from pyspark.sql import SparkSession
from textblob import Word

def spellcheck(word):
    value = Word(word).spellcheck()[0]
    if (value[1] < 0.7):
        return f'no_word_{value[0]}'
    return value[0]


def process_features():
    raw_features = "model_features.csv"  # Should be some file on your system
    spark = SparkSession.builder.appName("FeatureApp").master("local[*]").getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    spark.udf.register("correct", spellcheck)

    features = spark.read.format("csv")\
    .option("header", "true")\
    .option("inferSchema", "true")\
    .load(raw_features)

    features.registerTempTable("features")

    processed_features = spark.sql('select correct(feature) from features')
    processed_features.write.format("csv").save("processed_features.csv")
    #processed_features.to_csv('processed_features.csv', index = False)

    spark.stop()


if __name__ == '__main__':
    import time
    start = time.time()
    process_features()
    print(f'Seconds elapsed: {time.time() - start}')