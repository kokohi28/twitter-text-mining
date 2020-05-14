# local module
from twitter_scraper import get_tweets
import const as CONST
import nerExtractor as nex

# common
import os
import time
import math
from os import path
from datetime import datetime

# NLP
import nltk
import re
import string
# import Sastrawi
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer

# Dataframe, Numpy
import pandas as pd
import numpy as np

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from collections import Counter

# Blob
# import textblob
# from textblob import TextBlob

# Graph plot
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Constanta
TWEET_ONE_PAGE = 20
TOP_WORD = 40
MAX_PREVIEW_SCRAPPER = 70 # char
K = 5
PERCENT_TRAIN = 80

# VAR
# create stopwords
listStopword = set(stopwords.words('english'))
# Special stop-chars
additionalStopword = {'"', '\'', '\'', '’', '–', '…', '—', '\"', '”', '“', '•'}

# Lemmatizer
lemmatizer = WordNetLemmatizer()

def welcomeMessage():
  print('######################################################################')
  print('##                                                                  ##')
  print('##  NEWS ACCOUNT                                                    ##')
  print('##  TWITTER TEXT MINING                                             ##')
  print('##                                                                  ##')
  print('##  BY : - Koko Himawan Permadi (19051204111)                       ##')
  print('##       - M. Khafidhun Alim Muslim (17051204063)                   ##')
  print('##                                                                  ##')
  print('######################################################################')
  return

# Utilitas untuk get current millis
def localMillis():
  return int(round(time.time() * 1000))

# Print tweet with Ner (Named Entity Recognition)
def printTweet(i, tweetDate, content, ner):
  print('-----------------------------------------------------------------------')
  print(f'{i}. {tweetDate.strftime("%Y-%m-%d %H:%M:%S")}')
  # print with NER mapped
  if len(content) > MAX_PREVIEW_SCRAPPER:
    print(content[:MAX_PREVIEW_SCRAPPER] + '…')
  else:
    print(content)

  for j in range(0, nex.MONEY + 1):
    if len(ner[j]) > 0:
      if j == nex.PERSON:
        print(f'• Persons: {ner[j]}')
      elif j == nex.NORP:
        print(f'• Groups: {ner[j]}')
      elif j == nex.FAC:
        print(f'• Facilities: {ner[j]}')
      elif j == nex.ORG:
        print(f'• Organizations: {ner[j]}')
      elif j == nex.GPE:
        print(f'• Countries/Cities: {ner[j]}')
      elif j == nex.LOC:
        print(f'• Locations: {ner[j]}')
      elif j == nex.PRODUCT:
        print(f'• Objects: {ner[j]}')
      elif j == nex.EVENT:
        print(f'• Events: {ner[j]}')
      elif j == nex.WORK_OF_ART:
        print(f'• Artworks: {ner[j]}')
      elif j == nex.LAW:
        print(f'• Laws: {ner[j]}')
      elif j == nex.DATE:
        print(f'• Dates: {ner[j]}')
      elif j == nex.TIME:
        print(f'• Times: {ner[j]}')
      elif j == nex.MONEY:
        print(f'• Finances: {ner[j]}')
  return

# Process Token Ner
def processTokenNer(ner):
  # process ner
  tokenNer = []

  for j in range(0, nex.MONEY + 1):
    if len(ner[j]) > 0:
      if j == nex.PERSON: # Person as PERSON
        personNer = ner[j].split(',')
        for n in personNer:
          tokenNer.append(f'{n}/PERSON')
      elif j == nex.NORP: # Skip Organization
        pass #
      elif j == nex.FAC: # Facility as LOCATION
        facNer = ner[j].split(',')
        for n in facNer:
          tokenNer.append(f'{n}/LOCATION')
      elif j == nex.ORG:
        pass #
      elif j == nex.GPE: # City/Country as LOCATION
        gpeNer = ner[j].split(',')
        for n in gpeNer:
          tokenNer.append(f'{n}/LOCATION')
      elif j == nex.LOC: # Location as LOCATION
        locNer = ner[j].split(',')
        for n in locNer:
          tokenNer.append(f'{n}/LOCATION')
      elif j == nex.PRODUCT: # Skip Product
        pass #
      elif j == nex.EVENT: # Event as EVENT
        eventNer = ner[j].split(',')
        for n in eventNer:
          tokenNer.append(f'{n}/EVENT')
      elif j == nex.WORK_OF_ART: # Work of Art as PERSON
        woaNer = ner[j].split(',')
        for n in woaNer:
          tokenNer.append(f'{n}/PERSON')
      elif j == nex.LAW: # Skip Law
        pass #
      elif j == nex.DATE: # Date as DATE-TIME
        dateNer = ner[j].split(',')
        for n in dateNer:
          tokenNer.append(f'{n}/DATE-TIME')
      elif j == nex.TIME: # Time as DATE-TIME
        timeNer = ner[j].split(',')
        for n in timeNer:
          tokenNer.append(f'{n}/DATE-TIME')
      elif j == nex.MONEY: # Skip Money
        pass #

  return tokenNer

# Test tweet cluster and print
def testTweetCluster(vectorizer, model, testTweets, testTweetsDate, trainingDataLen):
  testClusterSize = {}

  i = 0
  for tweet in testTweets:
    test = vectorizer.transform([tweet])
    cluster = model.predict(test)

    print('-----------------------------------------------------------------------')
    print(f'{i + trainingDataLen}. {testTweetsDate[i].strftime("%Y-%m-%d %H:%M:%S")}')
    if len(tweet) > MAX_PREVIEW_SCRAPPER:
      print(tweet[:MAX_PREVIEW_SCRAPPER] + '…')
    else:
      print(tweet)

    clusterVal = cluster[0]
    print(f'• Cluster - {clusterVal}')

    # count and increment member cluster size
    if clusterVal in testClusterSize:
      newVal = testClusterSize[clusterVal]
      testClusterSize[clusterVal] = newVal + 1
    else:
      testClusterSize[clusterVal] = 1

    i = i + 1

  return testClusterSize

# Process K-Mean clustering
def processClustering(k, df):
  # Get tweet content, date
  allTweets = []
  allTweetsDate = []
  for index, row in df.iterrows():
    # remove punctuation, number
    content = row['content']
    # content = re.sub(r'\d+', '', content)
    cleanContent = content.translate(content.maketrans('', '', string.punctuation))
    allTweets.append(cleanContent)    
    allTweetsDate.append(row['date'])

  # Split training and test 
  trainingDataLen = math.ceil((len(df) * PERCENT_TRAIN) / 100)
  trainTweets = allTweets[:trainingDataLen] 
  testTweets = allTweets[trainingDataLen:]
  testTweetsDate = allTweetsDate[trainingDataLen:]

  # Create training data
  vectorizer = TfidfVectorizer(stop_words='english')
  trainData = vectorizer.fit_transform(trainTweets)

  # Create K-Mean object
  model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)

  # Trai the data
  model.fit(trainData)

  # Get Centroid
  orderCentroids = model.cluster_centers_.argsort()[:, ::-1]
  terms = vectorizer.get_feature_names()

  trainingClusterSize = dict(Counter(model.labels_))
  print(f'Size of training cluster member : {trainingClusterSize}')

  # Check top terms
  print('\nTop terms per cluster:')
  for i in range(k):
    print(f'\nCluster - {i} [size={trainingClusterSize[i]}]')
    countPreview = int(math.ceil(TOP_WORD / k))
    topTerms = []
    [ topTerms.append(terms[ind]) for ind in orderCentroids[i, :countPreview] ]
    print(topTerms)

  # Test the data
  print('\nTest the tweets cluster:')
  testClusterSize = testTweetCluster(vectorizer, model, testTweets, testTweetsDate, trainingDataLen + 1)
  print(f'\nSize of test cluster member : {testClusterSize}')

  # Merge training and test cluster member size
  clusterSize = {}
  # training
  for key in trainingClusterSize:
    clusterKey = f'Cluster - {key}'
    val = trainingClusterSize[key]

    if clusterKey in clusterSize:
      currVal = clusterSize[clusterKey]
      clusterSize[clusterKey] = currVal + val
    else:
      clusterSize[clusterKey] = val

  # test
  for key in testClusterSize:
    clusterKey = f'Cluster - {key}'
    val = testClusterSize[key]

    if clusterKey in clusterSize:
      currVal = clusterSize[clusterKey]
      clusterSize[clusterKey] = currVal + val
    else:
      clusterSize[clusterKey] = val

  print('Size of merged cluster member :')
  for key in clusterSize:
    print(f'- {key} : {clusterSize[key]}')

  return clusterSize

# Get Data from Twitter or Read from buffered/created CSV
def getData(username, count):
  # Export global var
  global listStopword
  global additionalStopword

  # Try read CSV
  fileCsvExist = path.exists(f'{username}.csv')
  if fileCsvExist:
    # read from CSV
    print('read from CSV')
    dateParse = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    df = pd.read_csv(f'{username}.csv', header='infer', parse_dates=['date'], date_parser=dateParse)
    df = df.sort_values(by='date')

    print(f'got {len(df)} tweets from @{username} from CSV')
    
    # define token
    tokenAll = [] # token all, alternative for set
    tokenNer = []

    # loop per tweet
    print('\nfound:')
    i = 1
    for index, row in df.iterrows():
      # print(index, row['date'], row['content'])

      # NER TAG
      doc, ner = nex.getDocNer(row['content'])

      # remove punctuation, number and to lowercase
      noNumbTweetContent = re.sub(r'\d+', '', row['content'])
      cleanTweetContent = noNumbTweetContent.translate(noNumbTweetContent.maketrans('', '', string.punctuation))
      
      # tokenize
      token = nltk.tokenize.word_tokenize(cleanTweetContent.lower())
      for t in token:
        if t not in listStopword \
           and t not in additionalStopword:
          tokenAll.append(lemmatizer.lemmatize(t))

      # fill token ner
      tokenNer = tokenNer + processTokenNer(ner)

      # print tweet
      printTweet(i, row['date'], row['content'], ner)
      i = i + 1
      print('')

    # create dataframe all token
    print('Calculate Frequency Distributions…')
    dfToken = pd.DataFrame(columns=['token', 'freq'])
    fd = FreqDist(tokenAll)
    for f in fd:
      entry = {'token' : f, 'freq' : fd[f]}
      dfToken = dfToken.append(entry, ignore_index=True)

    # create dataframe for ner
    dfNer = pd.DataFrame(columns=['ner', 'type', 'freq'])
    fdNer = FreqDist(tokenNer)
    for f in fdNer:
      entryNer = f.split('/')
      entry = {'ner' : entryNer[0], 'type' : entryNer[1], 'freq' : fdNer[f]}
      dfNer = dfNer.append(entry, ignore_index=True)

    return df, dfToken, dfNer
    
  else:
    # get from Twitter
    print(f'try get {count} tweets from @{username}')
    page = int(round(count / TWEET_ONE_PAGE))

    # create dataframe
    df = pd.DataFrame(columns=['date', # Date tweet
                               'year',
                               'month',
                               'day',
                               'ner_person', # PERSON : People, including fictional.
                               'ner_norp', # NORP : Nationalities or religious or political groups.
                               'ner_fac', # FAC : Buildings, airports, highways, bridges, etc.
                               'ner_org', # ORG : Companies, agencies, institutions, etc.
                               'ner_gpe', # GPE : Countries, cities, states.
                               'ner_loc', # LOC : Non-GPE locations, mountain ranges, bodies of water.
                               'ner_product', # PRODUCT : Objects, vehicles, foods, etc. (Not services.)
                               'ner_event', # EVENT : Named hurricanes, battles, wars, sports events, etc.
                               'ner_work_of_art', # WORK_OF_ART : Titles of books, songs, etc.
                               'ner_law', # LAW : Named documents made into laws.
                               'ner_date', # DATE : Absolute or relative dates or periods.
                               'ner_time', # TIME : Times smaller than a day.
                               'ner_money', # MONEY : Monetary values, including unit.
                               'content'] # Original tweet content
                               )

    # define token
    tokenAll = [] # token all, alternative for set
    tokenNer = []

    # loop per fetched tweet
    print('\nfound:')
    i = 1
    for t in get_tweets(username, pages=page):
      tweetDate = t[CONST.IDX_DATE]
      tweetContent = t[CONST.IDX_TWEET]
      respReplies = t[CONST.IDX_REPLIES]
      respRetweet = t[CONST.IDX_RETWEET]
      respLikes = t[CONST.IDX_LIKES]

      # Remove any links and numbers
      noLinkTweetContent = tweetContent.replace('pic.twitter.com', 'http://pic.twitter.com')
      noLinkTweetContent = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', noLinkTweetContent)
      noNumbTweetContent = re.sub(r'\d+', '', noLinkTweetContent)

      # remove punctuation
      cleanTweetContent = noNumbTweetContent.translate(noNumbTweetContent.maketrans('', '', string.punctuation))

      # NER TAG
      doc, ner = nex.getDocNer(noLinkTweetContent)

      # tokenize
      token = nltk.tokenize.word_tokenize(cleanTweetContent.lower())
      for t in token:
        if t not in listStopword \
           and t not in additionalStopword:
          tokenAll.append(lemmatizer.lemmatize(t))

      # fill token ner
      tokenNer = tokenNer + processTokenNer(ner)

      # add to dataframe
      entryDf = {'date': tweetDate,
                 'year': int(tweetDate.year),
                 'month': int(tweetDate.month),
                 'day': int(tweetDate.day),
                 'ner_person': ner[nex.PERSON],
                 'ner_norp': ner[nex.NORP],
                 'ner_fac': ner[nex.FAC],
                 'ner_org': ner[nex.ORG],
                 'ner_gpe': ner[nex.GPE],
                 'ner_loc': ner[nex.LOC],
                 'ner_product': ner[nex.PRODUCT],
                 'ner_event': ner[nex.EVENT],
                 'ner_work_of_art': ner[nex.WORK_OF_ART],
                 'ner_law': ner[nex.LAW],
                 'ner_date': ner[nex.DATE],
                 'ner_time': ner[nex.TIME],
                 'ner_money': ner[nex.MONEY],
                 'content': cleanTweetContent.lower()
                }
      df = df.append(entryDf, ignore_index=True)

      # print tweet
      printTweet(i, tweetDate, noLinkTweetContent, ner)
      i = i + 1
      print('')

    # print(df.head())

    # Save to CSV as buffer data
    if len(df) > 0:
      df = df.sort_values(by='date', ascending=False)
      df.to_csv(f'{username}.csv')

    # create dataframe all token
    print('Calculate Frequency Distributions…') 
    dfToken = pd.DataFrame(columns=['token', 'freq'])
    fd = FreqDist(tokenAll)
    for f in fd:
      entry = {'token' : f, 'freq' : fd[f]}
      dfToken = dfToken.append(entry, ignore_index=True)

    # create dataframe for ner
    dfNer = pd.DataFrame(columns=['ner', 'type', 'freq'])
    fdNer = FreqDist(tokenNer)
    for f in fdNer:
      entryNer = f.split('/')
      entry = {'ner' : entryNer[0], 'type' : entryNer[1], 'freq' : fdNer[f]}
      dfNer = dfNer.append(entry, ignore_index=True)

    return df, dfToken, dfNer

# MAIN PROGRAM
if __name__ == '__main__':
  welcomeMessage()

  # Input
  print('')
  try:
    userVal = ''
    countVal = 0
    k = 1

    # CONST.DEBUG = True
    if CONST.DEBUG:
      # Consider: SCMPNews vicenews AJEnglish AJENews BBCWorld guardiannews MetroUK
      #           cnni CNBC WIONews
      userVal = 'WIONews'
      countVal = 20
      k = K
    else:
      # Username and count tweet
      userVal = input("Input username news twitter: ")
      
      # Check the CSV
      fileCsvExist = path.exists(f'{userVal}.csv')
      if not fileCsvExist:
        countVal = input("Count of tweet to fetch (min. 20): ")

      # K
      getK = True
      while getK:
        kVal = input("Input K: ")
        if kVal.isnumeric():
          k = int(kVal)
          if k <= 1:
            print('K must be larger than 1... (Press any key to continue)')
            input('')
          else:
            # Proceed
            getK = False
        else:
          print('Invalid K... (Press any key to continue)')
          input('')

      print('')

    try:
      count = int(countVal)
      if count < TWEET_ONE_PAGE:
        count = TWEET_ONE_PAGE

      df, dfToken, dfNer = getData(userVal, count)
      if (len(df) <= 0):
        print('NO DATA FOUND, Exiting now!!!')
      else:
        print('\nProcessing clusters…')
        
        # Process clustering 
        clusterSize = processClustering(k, df)
        
        print('\nProcessing graph…')

        # sort dataframe first
        df.sort_values(by='date')
      
        # Get minimum/maximum date of dataframe
        dtMin = df.loc[df.index.min(), 'date']
        dtMax = df.loc[df.index.max(), 'date']

        # prepare token dataframe
        dfToken = dfToken.sort_values(by='freq', ascending=False)
        dfTokenShow = dfToken.head(TOP_WORD)

        # define view list
        viewList = ['TOKEN', 'CLUSTER', 'PERSON', 'LOCATION', 'DATE-TIME', 'EVENT']

        # Visualize
        title = f'Mining @{userVal} tweet from {dtMin.strftime("%Y-%m-%d  %H:%M:%S")} - {dtMax.strftime("%Y-%m-%d  %H:%M:%S")}'
        fig, ax = plt.subplots(num=title)
        plt.subplots_adjust(bottom=0.2)

        bnext = None
        listIdx = 1

        # Button Next event
        def next(event):
          # Export global var
          global k
          global clusterSize
          global plt
          global ax
          global bnext
          global listIdx
          global viewList
          global dfTokenShow
          global dfNer

          # set current
          currentList = viewList[listIdx]
          print(f'\nProcessing - {currentList}')

          # Clear graph
          ax.clear()

          # Token view
          if currentList == 'TOKEN':
            print(dfTokenShow.head())

            # Re-plot, Add graph info
            ax.set_title(f'{TOP_WORD} highest freq token from @{userVal}')
            ax.set_xlabel('freq', fontsize=11)
            ax.set_ylabel('token', fontsize=11)
            ax.tick_params(axis='both', which='major', labelsize=9)
            ax.tick_params(axis='both', which='minor', labelsize=7)

            # plot data        
            ax.barh(dfTokenShow['token'], dfTokenShow['freq'], align='center', color=np.random.rand(TOP_WORD, 3))
            ax.invert_yaxis()  # labels read top-to-bottom

          # Cluster view
          elif currentList == 'CLUSTER':
            # Re-plot, Add graph info
            ax.set_title(f'Cluster member size with K={k}')
            ax.set_xlabel('size', fontsize=11)
            ax.set_ylabel('cluster', fontsize=11)
            ax.tick_params(axis='both', which='major', labelsize=9)
            ax.tick_params(axis='both', which='minor', labelsize=7)

            # plot data
            xdata = [ key for key in clusterSize ]
            ydata = [ clusterSize[key] for key in clusterSize ]
            ax.barh(xdata, ydata, align='center', color=np.random.rand(TOP_WORD, 3))
            ax.invert_yaxis()  # labels read top-to-bottom

          # Ner view
          else:
            # filter dataframe Ner
          
            mask = dfNer['type'] == currentList
            ner = dfNer.loc[mask]
            ner = ner.head(TOP_WORD)
            print(ner.head())

            # Re-plot, Add graph info
            ax.set_title(f'highest freq token for {currentList}')
            ax.set_xlabel('freq', fontsize=11)
            ax.set_ylabel('ner', fontsize=11)
            ax.tick_params(axis='both', which='major', labelsize=9)
            ax.tick_params(axis='both', which='minor', labelsize=7)

            # plot data        
            ax.barh(ner['ner'], ner['freq'], align='center', color=np.random.rand(TOP_WORD, 3))
            ax.invert_yaxis()  # labels read top-to-bottom

          # Button label for next
          listIdx = listIdx + 1
          if listIdx >= len(viewList):
            listIdx = 0
          bnext.label.set_text(viewList[listIdx])

          # Plot
          ax.draw(renderer=None, inframe=False)
          plt.pause(0.0001)

        # Create button Predict
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, viewList[1]) # PERSON NEXT
        bnext.on_clicked(next)

        # Add graph info
        ax.set_title(f'{TOP_WORD} highest freq token from @{userVal}')
        ax.set_xlabel('freq', fontsize=11)
        ax.set_ylabel('token', fontsize=11)
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.tick_params(axis='both', which='minor', labelsize=7)

        # plot data        
        ax.barh(dfTokenShow['token'], dfTokenShow['freq'], align='center', color=np.random.rand(TOP_WORD, 3))
        ax.invert_yaxis()  # labels read top-to-bottom

        # finally show graph
        plt.show()

    except Exception as ex:
      print('Fail to process scrapping, cause:')
      raise(ex)
  except KeyboardInterrupt:
    pass
  
  print('\nExiting Now ...')
  