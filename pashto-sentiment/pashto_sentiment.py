
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, pipeline
from pyspark.ml.feature import IDF, CountVectorizer,StringIndexer, RegexTokenizer,StopWordsRemover
from pyspark.sql.functions import explode, regexp_replace, col, lit, reverse, split
from pyspark.sql.types import StringType,IntegerType
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# from pyspark.ml.base import PredictionModel

# import pandas as pd
# import numpy as np


    # df = spark.createDataFrame([(1, )], ['text'])
    # df.show()
    # numbers = [1, 2, 3]
    # df1 = spark.createDataFrame([(value,) for value in numbers], ['text'])
    # df1.show()
    # df3 = df.union(df1)
    # df3.show()
    # exit()

# sc = SparkContext(master="local[2")
# print(sc)
def remove_noise(row):
    print(row)
    return row.split()

def custom_input():
    # custom = "زه تاسره تا نکووم."
    inputs = input("Enter custom text: ")
    custom = spark.createDataFrame([(inputs, )], ["text"])
    custom.show()

    # Removing Digits
    custom = custom.select("text", split(custom.text, " ").alias("words"))
    log_custom_prediction = logistic_model.transform(custom)
    naive_custom_prediction = naive_model.transform(custom)

    print("Logistic Result: ")
    log_custom_prediction.show()

    print("Naive Result: ")
    naive_custom_prediction.show()

    
if __name__ == "__main__":
    print("Hello, World")

    # Create spark session 
    spark = SparkSession.builder.appName("Sentiment Analysis").getOrCreate()


    
    # Load the positive tweets data from nltk
    positive = spark.read.schema("col0 INT, col1 STRING").csv("file:///home/hayat/Desktop/Data Streaming Pipeline/pashto_tweets/positive-translated.csv").select("col1").cache()
    negative = spark.read.schema("col0 INT, col1 STRING").csv("file:///home/hayat/Desktop/Data Streaming Pipeline/pashto_tweets/negative-translated.csv").select("col1").cache()
    print(positive.columns)
    print(negative.columns)

    # Rename the column
    positive = positive.selectExpr("col1 as text")
    negative = negative.selectExpr("col1 as text")

    # reverse the text for pashto data.
    positive = positive.select(reverse(positive.text).alias("data"))
    negative = negative.select(reverse(negative.text).alias("data"))
   
    positive_cleaned = []
    negative_cleaned = []

    i = 1
    for token in positive.collect():
        print(i)
        positive_cleaned.append(remove_noise(token.data))
        i = i + 1
    i = 1
    for token in negative.collect():
        print(i)
        negative_cleaned.append(remove_noise(token.data))
        i = i + 1

    
    
    positive = spark.createDataFrame([(value, )for value in positive_cleaned], ['words'])
    negative = spark.createDataFrame([(value, )for value in negative_cleaned], ['words'])
    
    positive.show()
    # Add colum sentiment with value of (1) which is mean positive
    positive = positive.withColumn("sentiment", lit("Positive"))
    negative = positive.withColumn("sentiment", lit("Negative"))

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
    # positive = positive.withColumn("only_str", regexp_replace(col('text'), '\d+', ''))
    # negative = negative.withColumn("only_str", regexp_replace(col('text'), '\d+', ''))
    # positive.show(5)

    # Stage for pipeline
    # 1)  Tokenizer
    # 2)  StopWordsRemover
    # 3)  CountVectorizer
    # 4)  IDF (for logistic regrission)

    # regexp_tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
    # stopword_remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    count_vectorizer = CountVectorizer(inputCol="words", outputCol="features")

    # This is used for Logistic Regeresion
    idf = IDF(inputCol="features", outputCol="vectorizedFeatures")

    # Label Encoding/ String indexer

    # string_indexer = StringIndexer(inputCol="sentiment", outputCol="label")
    # string_indexer = string_indexer.fit(positive)
    # string_indexer.transform(positive).show(5)

    positive = positive.withColumn("label", lit(1))
    negative = negative.withColumn("label", lit(0))

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
    logistic_pipeline = Pipeline(stages=[  count_vectorizer, idf, logistic_re])
    naive_pipeline = Pipeline(stages=[ count_vectorizer, idf, naive_bayes])

    # print("Pipeline stages: ", pipeline.stages)

    # Build and train the logistic model
    logistic_model = logistic_pipeline.fit(trainDF)
    naive_model = naive_pipeline.fit(trainDF)

    # Predict and test the logistic model
    logistic_prediction = logistic_model.transform(dataset=testDF)
    naive_prediction = naive_model.transform(testDF)

    print("Logistic Prediction reuslt: ")
    logistic_prediction.select("words", "label", "prediction").show(5)

    print("Naive Bayes prediction result: ")
    naive_prediction.select( "words", "label", "prediction").show(5)

    # Evaluation of Model
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="accuracy")
    logistic_accuracy = evaluator.evaluate(logistic_prediction)
    naive_accuracy = evaluator.evaluate(naive_prediction)

    print("The Logistic Prediction accuracy is: ", logistic_accuracy)
    print("The Naive Prediction accuracy is: ", naive_accuracy)

    
    print("\n==================== Single Predition of Custo ==============\n")
    # Making sigle prediction on Custom data
    
    # print("\n\n============================ Custom input analysis ===================")
    # custom_input()

    # try_again = input("\nWould you like to continue? (هو/نه)")

    # while try_again == 'هو':
    #     # custom = input("Enter custom text: ")
    #     # custom_tokenized = word_tokenize(text=custom)
    #     # print("Custom tweet is: ", custom)
    #     # custom_data = dict([token, True] for token in custom_tokenized)
    #     # print("The prediction is => ", model.classify(custom_data))
    #     #
    #     custom_input()
    #     try_again = input("\nWould you like to continue? (هو/نه)")




    # custom = input("Enter data to predict: ")

    custom = spark.createDataFrame([("زه به د ووژنم", )], ["text"])
    custom.show()

    # Removing Digits
    custom = custom.select("text", split(custom.text, " ").alias("words"))
    log_custom_prediction = logistic_model.transform(custom)
    naive_custom_prediction = naive_model.transform(custom)

    print("Logistic Result: ")
    log_custom_prediction.show()

    print("Naive Result: ")
    naive_custom_prediction.show()


    exit()
    logistic_model.save("/home/hayat/logistic")
    naive_model.save("/home/hayat/naive")

    import pickle
    pickle.dump(naive_model, open(file="nb_model.sav", mode="wb"))
    pickle.dump(logistic_model, open(file="lg_model.sav", mode="wb"))

    exit()
    # Save the model
    path = "/home/hayat/Desktop/Trained Models/pyspark_logistic_regrission_aug_29_2021"
    logistic_model.save(path=path)

    # Load the trained model back 
    # from pyspark.ml.pipeline import PipelineModel
    # model = PipelineModel.load(path=path)

