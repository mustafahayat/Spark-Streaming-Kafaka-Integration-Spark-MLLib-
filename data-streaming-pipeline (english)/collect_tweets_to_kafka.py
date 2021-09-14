#!/usr/bin/python3 


# My Twitter Api Keys and Tokens
import twitter_api as conf
from kafka import KafkaProducer
from datetime import datetime
import tweepy
import sys
import re


TWEET_TAGS = ['taliban', 'TalibanOurGaurdians']

KAFKA_BROKER = 'localhost:9092'
KAFKA_TOPIC = 'taliban_tweets'


class Streamer(tweepy.StreamListener):

    def on_error(self, status_code):
        if status_code == 402:
            return False

    def on_status(self, status):
        tweet = status.text

        tweet = re.sub(r'RT\s@\w*:\s', '', tweet)
        tweet = re.sub(r'https?.*', '', tweet)

        global producer
        producer.send(KAFKA_TOPIC, bytes(tweet, encoding='utf-8'))

        d = datetime.now()

        print(f'[{d.hour}:{d.minute}.{d.second}] sending tweet')


# put your Twitter API keys here
consumer_key = conf.consumer_key
consumer_secret_key = conf.consumer_secret

access_token = conf.access_token
access_token_secret = conf.access_token_secret

auth = tweepy.OAuthHandler(consumer_key, consumer_secret_key)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

streamer = Streamer()
stream = tweepy.Stream(auth=api.auth, listener=streamer)

try:
    producer = KafkaProducer(bootstrap_servers=KAFKA_BROKER)
except Exception as e:
    print(f'Error Connecting to Kafka --> {e}')
    sys.exit(1)

stream.filter(track=TWEET_TAGS)