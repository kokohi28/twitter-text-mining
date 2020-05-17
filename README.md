# Twitter news text mining
Text mining news tweet from news account Twitter. For most frequent term, NER (Named entity recognition) extraction, and clustering.

Twitter scaper module modified from :
https://github.com/helmisatria/twitter-scraper.git

## Note
Specific for english tweet

## Preview
Most frequent term
![term](https://github.com/kokohi28/twitter-text-mining/blob/master/token_sample.png?raw=true)

Ner Person frequent
![ner_person](https://github.com/kokohi28/twitter-text-mining/blob/master/ner_person_sample.png?raw=true)

## Requirements
* Python 3.7

## Requirements Library
* numpy ->
  $ pip install numpy

* pandas ->
  $ pip install pandas

* nltk ->
  $ pip install nltk

* sklearn ->
  $ pip install scikit-learn

* spacy ->
  $ pip install spacy

* spacy english model ->
  $ python -m spacy download en_core_web_sm

* requests-html ->
  $ pip install requests-html

* matplotlib ->
  $ pip install matplotlib

## File Structure
### py files
* const.py -> Constant for all python project files

* nerExtractor.py -> Extract ner information from tweet

* twitter_scraper.py -> Twitter scraper

* main.py -> Main program

## How to Run
$ python3 main.py
