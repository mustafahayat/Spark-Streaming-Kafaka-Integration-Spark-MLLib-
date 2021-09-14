#!/usr/bin/python3                                                                                                      
                                                                                                                        
from pyspark import SparkContext                                                                                        
from pyspark.sql import SparkSession                                                                                    
from pyspark.streaming import StreamingContext                                                                          
from pyspark.streaming.kafka import KafkaUtils 
from pyspark.streaming.kafka import KafkaUtils
from pyspark.ml.pipeline import PipelineModel

KAFKA_TOPIC = "tweets"
HIVE_DATABASE = "default"
HIVE_TALBE = "tweets"

def handle_rdd(rdd):                                                                                                    
    if not rdd.isEmpty():                                                                                               
        global spark                                                                                                       
        df = spark.createDataFrame(rdd, schema=['text', 'words', 'length'])                                                
        df.show()                                                                                                       
        df.write.saveAsTable(name='default.tweets', format='hive', mode='append')                                       

if __name__ == "__main__":
                                                                                                                        
    spark_context = SparkContext(appName="Spark Streaming")   

    # this will chech the Streaming every five seconds
    streaming_context = StreamingContext(spark_context, 5)                                                                                           
                                                                                                                            
    spark = (
    SparkSession.builder
    .appName("Spark Streaming")

    .config("spark.sql.warehouse.dir", "/user/hive/warehouse")
    .config("hive.metastore.uris", "thrift://localhost:9083")
    .enableHiveSupport()
    .getOrCreate() 
    )

    # Setting log level to warning                                                                                                                       
    spark.sparkContext.setLogLevel('WARN')                                                                                     
                                                                         
    ks = KafkaUtils.createDirectStream(ssc=streaming_context, topics=[KAFKA_TOPIC], kafkaParams= {'metadata.broker.list': 'localhost:9092'})                       
    print("======================") 
    print(ks)
    print("====================")
    lines = ks.map(lambda x: x[1])                                                                                          

    print("============Lines==============") 
    print(lines.count())
    print("================================")

    transform = lines.map(lambda tweet: (tweet, int(len(tweet.split())), int(len(tweet))))                                  

    print("==============Tranform================")
    print(transform)
    print("==================================")                                                                                                  
    transform.foreachRDD(handle_rdd)                                                                                        
                                                                                                                            
    streaming_context.start()                                                                                                             
    streaming_context.awaitTermination()