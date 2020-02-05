
import numpy as np
import pandas as pd

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from textblob import TextBlob, Word

from pyspark.sql import SparkSession

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')
#english_corpus = set(words.words())


def tokenize(text):
    text = str(TextBlob(text).lower())
    text = str(tb.correct())
    text = re.sub(r"[^a-z]", " ", text)
    word_list = word_tokenize(text)
    word_list = [w for w in word_list if w not in stopwords.words("english")]
    word_list = [WordNetLemmatizer().lemmatize(w, pos='v') for w in word_list]
    #word_list = set(word_list).intersection(english_corpus)
    
    return word_list
  


def main():
    spark = SparkSession.builder.appName("NlpApp").getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    spark.udf.register("tokenize", tokenize)

    dataset = spark\
    .read\
    .option("inferSchema", "true")\
    .option("header", "true")\
    .csv("data/disaster_messages.csv")

    dataset.createOrReplaceTempView("disaster_messages")

    preprocessed = spark.sql("select tokenize(message) from disaster_messages").toPandas()
    preprocessed.to_csv('disaster_messages_processed.csv', index = False)

if __name__ == '__main__':
    main()