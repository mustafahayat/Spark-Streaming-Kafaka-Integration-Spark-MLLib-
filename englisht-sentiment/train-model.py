
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, pipeline
from pyspark.ml.feature import IDF, CountVectorizer,StringIndexer, RegexTokenizer,StopWordsRemover
from pyspark.sql.functions import regexp_replace, col, lit
from pyspark.sql.types import StringType,IntegerType
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

from pyspark.ml.base import PredictionModel

# import pandas as pd
# import numpy as np


# sc = SparkContext(master="local[2")
# print(sc)

print("Hello, World")

# Create spark session 
spark = SparkSession.builder.appName("Sentiment Analysis").getOrCreate()

# Load the positive tweets data from nltk
positive =  spark.read.json("file:///home/hayat/Desktop/text sentiment with spark/englisht-sentiment/twitter_samples/positive_tweets.json").select("text").cache()
negative =  spark.read.json("file:///home/hayat/Desktop/text sentiment with spark/englisht-sentiment/twitter_samples/negative_tweets.json").select("text").cache()



# Add colum sentiment with value of (1) which is mean positive
positive = positive.withColumn("sentiment", lit("Positive"))
negative = negative.withColumn("sentiment", lit("Negative"))

print(positive.printSchema())

# Show the count of the data
print("The count of positive is: ", positive.count())
print("The count of negative is: ", negative.count())

print("Positive data: ")
positive.show(5)

print("Negative data: ")
negative.show(5)

# Drop the row which contain the null vlaue
# positive = positive.dropna()
# print("The count is: ", positive.count())

# Clean the text from digits
positive = positive.withColumn("text", regexp_replace(col('text'), '\d+', ''))
negative = negative.withColumn("text", regexp_replace(col('text'), '\d+', ''))
positive.show(5)

# Stage for pipeline
# 1)  Tokenizer
# 2)  StopWordsRemover
# 3)  CountVectorizer
# 4)  IDF (for logistic regrission)

regexp_tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
stopword_remover = StopWordsRemover(inputCol="words", outputCol="filtered")
count_vectorizer = CountVectorizer(inputCol="filtered", outputCol="features")
idf = IDF(inputCol="features", outputCol="vectorizedFeatures")

# Label Encoding/ String indexer

# string_indexer = StringIndexer(inputCol="sentiment", outputCol="label")
# string_indexer = string_indexer.fit(positive)
# string_indexer.transform(positive).show(5)

# exit()

positive = positive.withColumn("label", lit(1))
negative = negative.withColumn("label", lit(0))

print("==================================")
dataset = positive.union(negative)

# import random
# random.shuffle(dataset)
dataset = dataset.sort("label", ascending=True)

print("Sorted Data =============>")
dataset.show()

# Split datasets to trian and test
(trainDF, testDF) = dataset.randomSplit([0.8, 0.2], seed=5000)

print("\n===== Traind Data =========\n")
trainDF.show(5)

print("\n===== Test Data =========\n")
testDF.show(5)

# Create Logistic Regrission Estimater
logistic_re = LogisticRegression(featuresCol="vectorizedFeatures", labelCol="label")
naive_bayes = NaiveBayes(modelType="multinomial", featuresCol="vectorizedFeatures", labelCol="label")
# Building a Pipeline
logistic_pipeline = Pipeline(stages=[regexp_tokenizer, stopword_remover, count_vectorizer, idf, logistic_re])
naive_pipeline = Pipeline(stages=[regexp_tokenizer, stopword_remover, count_vectorizer, idf, naive_bayes])

# print("Pipeline stages: ", pipeline.stages)

# Build and train the logistic model
logistic_model = logistic_pipeline.fit(trainDF)
naive_model = naive_pipeline.fit(trainDF)

# Predict and test the logistic model
logistic_prediction = logistic_model.transform(dataset=testDF)
naive_prediction = naive_model.transform(testDF)

print("Logistic Prediction reuslt: ")
logistic_prediction.select("text", "words", "label", "prediction").show(5)

print("Naive Bayes prediction result: ")
naive_prediction.select("text", "words", "label", "prediction").show(5)

# Evaluation of Model
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="accuracy")
logistic_accuracy = evaluator.evaluate(logistic_prediction)
naive_accuracy = evaluator.evaluate(naive_prediction)

print("The Logistic Prediction accuracy is: ", logistic_accuracy)
print("The Naive Prediction accuracy is: ", naive_accuracy)

print("\n==================== Single Predition of Custo ==============\n")
# Making sigle prediction on Custom data
custom = spark.createDataFrame([("I hate you", StringType())], ["custom"])
custom.show()

# Removing Digits
custom = custom.withColumn("text", regexp_replace(col('custom'), '\d+', ''))

log_custom_prediction = logistic_model.transform(custom)
naive_custom_prediction = naive_model.transform(custom)

print("Logistic Result: ")
log_custom_prediction.show()

print("Naive Result: ")
naive_custom_prediction.show()

print("Model saving ...")

logistic_model.write().overwrite().save("file:///home/hayat/Desktop/text sentiment with spark/englisht-sentiment/trained-model/logistic-regression")
naive_model.write().overwrite().save("file:///home/hayat/Desktop/text sentiment with spark/englisht-sentiment/trained-model/naive-bayes")

print("Successfully saved the model in the file")
# Load the trained model back 
# from pyspark.ml.pipeline import PipelineModel
# model = PipelineModel.load(path=path)

