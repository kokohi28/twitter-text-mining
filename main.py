# local module
from twitter_scraper import get_tweets
import const as CONST

# common
import os
import time
from os import path
from datetime import datetime

# NLP
import nltk
import re
import string
import Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

# Dataframe, Numpy
import pandas as pd
import numpy as np

# Blob
import textblob
from textblob import TextBlob

# Graph plot
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Constanta
TWEET_ONE_PAGE = 20
TOP_WORD = 30
TITLE = 'INDONESIAN POLITICAN TWITTER TEXT MINING'

def welcomeMessage():
  print('###################################################')
  print('#                                                 #')
  print('#  INDONESIAN POLITICAN TWITTER TEXT MINING       #')
  print('#                                                 #')
  print('#  BY : - M. Khafidhun Alim Muslim (17051204063)  #')
  print('#       - Koko Himawan Permadi (19051204111)      #')
  print('#                                                 #')
  print('###################################################')
  return

# Utilitas untuk get current millis
def localMillis():
  return int(round(time.time() * 1000))

def getData(username, count):
  #  Try read CSV
  fileCsvExist = path.exists(f'{username}.csv')
  if fileCsvExist:
    # read from CSV
    print('read from CSV')
    dateParse = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    df = pd.read_csv(f'{username}.csv', header='infer', parse_dates=['date'], date_parser=dateParse)
    df = df.sort_values(by='date')

    print('tweet from CSV: ', len(df))
    print(df)

    # create stopwords
    listStopword = set(stopwords.words('indonesian'))

    # tokenize
    # tokenMontly = [] # token per document
    tokenAll = [] # token all, alternative for set

    for index, row in df.iterrows():
      # print(index, row['date'], row['content'])
      token = nltk.tokenize.word_tokenize(row['content'])
      for t in token:
        if t not in listStopword:
          tokenAll.append(t)

    dfToken = pd.DataFrame(columns=['token', 'freq'])
    fd = FreqDist(tokenAll)
    for f in fd:
      entry = {'token' : f, 'freq' : fd[f]}
      dfToken = dfToken.append(entry, ignore_index=True)

    return df, dfToken
  else:
    # create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # create stopwords
    listStopword = set(stopwords.words('indonesian'))

    print('try get', count, 'tweets')
    page = int(round(count / TWEET_ONE_PAGE))

    df = pd.DataFrame(columns=['date', 'year', 'month', 'day', 'content', 'replies', 'retweet', 'likes'])

    # tokenize
    # tokenMontly = [] # token per document
    tokenAll = [] # token all, alternative for set

    for t in get_tweets(username, pages=page):
      tweetDate = t[CONST.IDX_DATE]
      tweetContent = t[CONST.IDX_TWEET]
      respReplies = t[CONST.IDX_REPLIES]
      respRetweet = t[CONST.IDX_RETWEET]
      respLikes = t[CONST.IDX_LIKES]

      # Remove any links and numbers
      tweetContent = tweetContent.replace('pic.twitter.com', 'http://pic.twitter.com')
      tweetContent = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweetContent)
      tweetContent = re.sub(r'\d+', '', tweetContent)

      # lower the string, remove punctuation
      tweetContent = tweetContent.lower().translate(tweetContent.maketrans('', '', string.punctuation))

      # stemming
      tweetContent = stemmer.stem(tweetContent)

      token = nltk.tokenize.word_tokenize(tweetContent)
      for t in token:
        if t not in listStopword:
          tokenAll.append(t)

      entryDf = {'date': tweetDate,
                 'year': int(tweetDate.year),
                 'month': int(tweetDate.month),
                 'day': int(tweetDate.day),
                 'content': tweetContent,
                 'replies': respReplies,
                 'retweet': respRetweet,
                 'likes': respLikes
                }
      df = df.append(entryDf, ignore_index=True)

    print('tweet found: ', len(df))
    print(df.head())

    if len(df) > 0:
      df.to_csv(f'{username}.csv')
    df = df.sort_values(by='date')

    dfToken = pd.DataFrame(columns=['token', 'freq'])
    fd = FreqDist(tokenAll)
    for f in fd:
      entry = {'token' : f, 'freq' : fd[f]}
      dfToken = dfToken.append(entry, ignore_index=True)

    return df, dfToken

# MAIN PROGRAM
if __name__ == '__main__':
  welcomeMessage()

  # Input
  print('')
  try:
    userVal = ''
    countVal = 0

    CONST.DEBUG = True
    if CONST.DEBUG:
      userVal = 'jokowi'
      countVal = 500
    else:
      userVal = input("Masukkan username twitter: ")
      countVal = input("Jumlah tweet yang akan diambil (minimal 20): ")
      print('')

    # try:
    count = int(countVal)
    if count < TWEET_ONE_PAGE:
      count = TWEET_ONE_PAGE

    df, dfToken = getData(userVal, count)
    if (len(df) <= 0):
      print('NO DATA FOUND, Exiting now!!!')
    else:
      print('Processing text')

      dfToken = dfToken.sort_values(by='freq', ascending=False)
      dfTokenShow = dfToken.head(TOP_WORD)

      # Visualize
      fig, ax = plt.subplots(num=TITLE)
      plt.subplots_adjust(bottom=0.2)

      # Add graph info
      ax.set_title(f'{TOP_WORD} token dengan freq tertinggi')
      ax.set_xlabel('token', fontsize=14)
      ax.set_ylabel('freq', fontsize=14)
      plt.xticks(rotation=90)
      # ax.grid(linestyle='-', linewidth='0.5', color='gray')
      
      # plot trained data
      ax.bar(dfTokenShow['token'], dfTokenShow['freq'], color=np.random.rand(TOP_WORD, 3))
      # ax.plot(dfTokenShow['token'], dfTokenShow['freq'])

      # finally show graph
      plt.show()

    # except Exception as ex:
    #   print('Fail to process scrapping, cause:')
    #   print(ex)
  except KeyboardInterrupt:
    pass
  
  print('\nExiting Now ...')
  