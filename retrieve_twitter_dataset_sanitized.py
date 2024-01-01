# first, install tweepy with the following command: conda install -c conda-forge tweepy

import tweepy
import json
import pandas as pd

from time import sleep

client = tweepy.Client("YOUR SECRET HERE", wait_on_rate_limit=True)

result = []; tweets = []; query="zurich -is:retweet lang:en"

for response in tweepy.Paginator(client.search_recent_tweets, query=query, tweet_fields=['entities'], user_fields=['name', 'id'], expansions=['author_id'], max_results=100, limit=2500): #for retrieving final dataset, use max_results=100 and limit=1000
    usersdict = {x.id:x.username for x in response.includes['users']}
    sleep(1)
    for tweet in response.data:
        tweets.append(tweet)
        result.append({'username': usersdict[tweet.author_id], 'text': tweet.text, 'entities': tweet.entities, 'context_annotations': tweet.context_annotations})
          
#store the data in a pandas dataframe            
df = pd.DataFrame(result)

# storing the data in JSON format
df.to_json('tweets_2500.json', orient = 'split', compression = 'infer', index = 'true')
# uncomment below line to additionally store the data in a JSONL file
# df.to_json('tweets_2500.jsonl', orient = 'records', lines=True)
