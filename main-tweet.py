from twitter_scraper_helmi import get_tweets
import os
import time
import csv
import re
import string
import Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

TWEET_ONE_PAGE = 20

def welcomeMessage():
  print('##############################################################################')
  print('####                                                                      ####')
  print('####   INDONESIAN POLITICAN TWITTER TEXT MINING                           ####')
  print('####                                                                      ####')
  print('####   BY : - M. Khafidhun Alim Muslim (17051204063)                      ####')
  print('####        - Koko Himawan Permadi (19051204111)                          ####')
  print('####                                                                      ####')
  print('##############################################################################')
  return

# Utilitas untuk get current millis
def localMillis():
  return int(round(time.time() * 1000))

def getData(username, count):
  data = []
  error = 0

  print('try get', count, 'tweets')
  page = int(round(count / TWEET_ONE_PAGE))

  for t in get_tweets(username, pages=page):
    data.append(t)
    if (t['status'] != 'ok'):
      error += 1

  print('tweet found: ', len(data))
  print('error count: ', error)
  return (data, error)

def processDataCSVWriter(data, username):
  # create stemmer
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()

  # Save to CSV for external use, ex : Orange
  with open('{}.csv'.format(username), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["no", "date", "tweet"])

    #### BENCHMARK
    totalMillis = 0
    startMillis = localMillis()
    #### BENCHMARK

    for i, t in enumerate(data):
      # Parse tweet date - content
      tweetDate = t['tweet']['time']
      tweetContent = t['tweet']['text']
    
      # Remove any links and numbers
      tweetContent = tweetContent.replace('pic.twitter.com', 'http://pic.twitter.com')
      tweetContent = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweetContent)
      tweetContent = re.sub(r'\d+', '', tweetContent)

      # lower the string, remove punctuation
      tweetContent = tweetContent.lower().translate(tweetContent.maketrans('', '', string.punctuation))

      # stemming
      tweetContent = stemmer.stem(tweetContent)

      idn = i + 1
      # print(str(idn) + '.' + tweetDate.strftime("%Y/%m/%d %H:%M:%S") + ' ' + tweetContent)
      writer.writerow([idn, tweetDate.strftime("%Y/%m/%d %H:%M:%S"), tweetContent])

    #### BENCHMARK
    endMillis = localMillis()
    totalMillis += (endMillis - startMillis)
    print('..')
    print('............ WRITE USE CSV WRITER TAKES: ' + str(endMillis - startMillis) + ' ms, total: ' + str(totalMillis) + ' ms')
    print('..')
    startMillis = endMillis
    #### BENCHMARK

  return

def processDataFileWriter(data, username):
  # create stemmer
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()

  # Save to CSV for external use, ex : Orange
  with open('{}.csv'.format(username), 'w', newline='') as file:
    file.write("no,date,tweet\n")

    #### BENCHMARK
    totalMillis = 0
    startMillis = localMillis()
    #### BENCHMARK

    row = ''
    for i, t in enumerate(data):
      # Parse tweet date - content
      tweetDate = t['tweet']['time']
      tweetContent = t['tweet']['text']
    
      # Remove any links and numbers
      tweetContent = tweetContent.replace('pic.twitter.com', 'http://pic.twitter.com')
      tweetContent = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweetContent)
      tweetContent = re.sub(r'\d+', '', tweetContent)

      # lower the string, remove punctuation
      tweetContent = tweetContent.lower().translate(tweetContent.maketrans('', '', string.punctuation))

      # stemming
      tweetContent = stemmer.stem(tweetContent)

      idn = i + 1
      # print(str(idn) + '.' + tweetDate.strftime("%Y/%m/%d %H:%M:%S") + ' ' + tweetContent)
      row = row + str(idn) + ',' + tweetDate.strftime("%Y/%m/%d %H:%M:%S") + ',' + tweetContent + '\n'
      if idn % 50 == 0:
        file.writelines(row)
        row = ''
    else:
      file.writelines(row)
      file.close

    #### BENCHMARK
    endMillis = localMillis()
    totalMillis += (endMillis - startMillis)
    print('..')
    print('............ WRITE USE FILE WRITER TAKES: ' + str(endMillis - startMillis) + ' ms, total: ' + str(totalMillis) + ' ms')
    print('..')
    startMillis = endMillis
    #### BENCHMARK

  return

# MAIN PROGRAM
if __name__ == '__main__':
  welcomeMessage()

  # Input
  print('')
  try:
    # inputVal = 'jokowi'
    userVal = input("Masukkan username twitter: ")
    countVal = input("Jumlah tweet yang akan diambil (minimal 20): ")
    print('')

    try:
      count = int(countVal)
      if count < TWEET_ONE_PAGE:
        count = TWEET_ONE_PAGE

      data, error = getData(userVal, count)
      if (len(data) <= 0):
        print('NO DATA FOUND, Exiting now!!!')
      else:
        print('Processing text')
        # processDataFileWriter(data, userVal)
        processDataCSVWriter(data, userVal)

    except Exception as ex:
      print('Fail to process scrapping, cause:')
      print(ex)
  except KeyboardInterrupt:
    pass
  
  print('\nExiting Now ...')
  