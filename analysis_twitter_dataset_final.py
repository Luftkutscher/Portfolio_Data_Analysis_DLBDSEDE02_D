# Various libs
import tweepy
import json
import sys
import collections
import re

# NLP libs
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# DF and Arrays
import numpy as np
import pandas as pd

# Gensim libs
import gensim
import gensim.corpora as corpora
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim import models
from gensim.utils import simple_preprocess

# Text color lib
from termcolor import colored

# Plotting libs
from wordcloud import WordCloud
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import matplotlib.pyplot as plt


# run only once, to download ntlk dataset
# nltk.download()

class color:
  PURPLE = '\033[95m'
  CYAN = '\033[96m'
  DARKCYAN = '\033[36m'
  BLUE = '\033[94m'
  GREEN = '\033[92m'
  YELLOW = '\033[93m'
  RED = '\033[91m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'
  END = '\033[0m'

        
""" HELPER FUNCTIONS BELOW """  

# Removes Emojis with the help of regular expressions
def remove_emoji(data):
    emoj = re.compile("["
        u"\U00002700-\U000027BF"  # Dingbats
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U00002600-\U000026FF"  # Miscellaneous Symbols
        u"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols And Pictographs
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

# Using TweetTokenizer from NLTK to tokenize data
def tokenize(sentences):
    for sentence in sentences:
        # Using yield instead of return enables the function to resume where left off
        yield(tk.tokenize(str(sentence)))

# Lemmatizing data using WordNetLemmatizer from NLTK
def lemmatize_text(df_text):
    lemmatized =[]
    for w in df_text:
        lemmatized.append(ltz.lemmatize(w))
    return lemmatized

# Function to detect common phrases, i.e. multi-word expressions, word n-gram collocations
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

# Function that searches the data for words beginning with a hashtag
def extract_hashtags(text):
    # Define the regular expression
    regex = "#(\w+)"
    # Extract the hashtags
    hashtag_list = re.findall(regex, text)
    for hashtag in hashtag_list:
        htag.append(hashtag)
        return htag

# Function to remove stopwords
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]
    

""" MAIN CODE BELOW """


# Load the dataset, i.e. reading the JSON file
df = pd.read_json('tweets_2477.json', orient ='split')

########################
#                      #
# Initial Sanitation   #
#                      #
######################## 

# Convert the text to lowercase
df['text'] = \
df['text'].map(lambda x: x.lower())

# Remove links
df['text'] = \
df['text'].map(lambda x: re.sub(r'http\S+', '', x, flags=re.MULTILINE))

# Remove emojis
df['text'] = \
df['text'].map(lambda x: remove_emoji(x))    
    
########################
#                      #
#   Find the Hashtags  #
#                      #
########################


# search and print 10 most common hashtags
print(color.RED +"\nThese are the 10 most used #hashtags:\n" + color.END)
htag = []

# Loop through the dataframe, extact the hashtags and count them
for ind in df.index:
    extract_hashtags(df['text'][ind])
    hashtags = collections.Counter()
    hashtags.update(htag)

# Print the 10 most common hashtags, counted by collections.Counter()
for a,b in hashtags.most_common(10):
    print("{}: {}".format(a, b))

########################
#                      #
#       Top Users      #
#                      #
########################        

# search and print 10 most active users
print(color.RED +"\nThese are the 10 most active users:\n" + color.END)
print(df['username'].value_counts().head(10))

  
########################
#                      #
#   Define Stopwords   #
#                      #
######################## 

# Add stopwords from NTLK, punctuation, integers and others
stop_words = stopwords.words('english')
stop_words += list(string.punctuation)
stop_words += ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
stop_words += ['rt', 'via', 'zurich'] # rt -> retweets, via -> original author

########################
#                      #
#     NLP / TF-IDF     #
#                      #
######################## 

# Initialize TweetTokenizer to strip handles and reduce length of recurring chars
tk = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

ltz = WordNetLemmatizer()

# Convert df to list, lemmatize, tokenize and remove stopwords
data = df.text.values.tolist()
data_words = lemmatize_text(data)
data_words = list(tokenize(data_words))
data_words = remove_stopwords(data_words)

# Create model and apply n-gram using gensim
bigram = models.Phrases(data_words, min_count=2, threshold=5)
bigram_mod = models.phrases.Phraser(bigram)
data = make_bigrams(data_words)

# Create dictionary, corpus and tf-idf vectors
dct = Dictionary(data)
corpus = [dct.doc2bow(line) for line in data_words]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]


########################
#                      #
#          LSA         #
#                      #
######################## 

lsi_model = models.LsiModel(corpus_tfidf, id2word=dct, num_topics=5)  # initialize an LSI transformation


########################
#                      #
#      Visualize       #
#                      #
######################## 

# Print Top 5 LSA-Topics with the 5 most prominent words for each
print(color.RED +"\nThese are the Top 5 discussed topics:\n" + color.END)

display(lsi_model.print_topics(num_topics=5, num_words=5))

print(color.RED +"\nResulting Wordclouds:\n" + color.END)

# Create a WordCloud for each topic
for t in range(lsi_model.num_topics):
    plt.figure()
    plt.imshow(WordCloud().fit_words(dict(lsi_model.show_topic(t, 50))))
    plt.axis("off")
    plt.title("Topic #" + str(t+1))
    # Uncomment below line to save as png
    # plt.savefig("Topic #" + str(t+1) + ".png", bbox_inches='tight')
    plt.show()

########################
#                      #
#     Bonus: W2V       #
#                      #
######################## 

print(color.RED +"\nWord Embedding using Word2Vec:\n" + color.END)

# Create Word2Vec model, including training and visualisation
w2v_model = gensim.models.Word2Vec(data, min_count=5, vector_size=200)
w2v_model.build_vocab(data)
w2v_model.train(data, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

# Create vocabulary, transform to 2D and visualize
words = list(w2v_model.wv.key_to_index)
X = [w2v_model.wv[word] for i, word in enumerate(words)]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
pca_df = pd.DataFrame(result, columns = ['x','y'])
pca_df['word'] = words
pca_df.head()

N = 1000000
fig = go.Figure(data=go.Scattergl(
   x = pca_df['x'],
   y = pca_df['y'],
   mode='markers',
   marker=dict(
       color=np.random.randn(N),
       colorscale='Viridis',
       line_width=1
   ),
   text=pca_df['word'],
   textposition="bottom center"
))

fig.update_layout(
    autosize=False,
    width=800,
    height=800,
)

# Uncomment below line to save as html
# fig.write_html("word2vec.html")
fig.show()