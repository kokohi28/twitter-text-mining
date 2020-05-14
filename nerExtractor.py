import spacy

'''
Install with pip
$ python3 -m pip install spacy

Download model
$ python3 -m spacy download en_core_web_sm
'''

# Spacy model "en_core_web_sm"
nlp = spacy.load("en_core_web_sm")

# See https://spacy.io/api/annotation
'''
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
'''

# LIST IDX
PERSON = 0
NORP = 1
FAC = 2
ORG = 3
GPE = 4
LOC = 5
PRODUCT = 6
EVENT = 7
WORK_OF_ART = 8
LAW = 9
DATE = 10
TIME = 11
MONEY = 12

# Get Mapped Ner (Named Entity Recognition)
def getNer(doc):
  tags = [(ent.text, ent.label_) for ent in doc.ents]

  # String TAG
  ner = []

  # Init with empty string
  # Get last idx ner
  for i in range(0, MONEY + 1):
    ner.append('')

  # parse tag to list with idx ner
  for t in tags:
    if t[1] == 'PERSON':
      ner[PERSON] = ner[PERSON] + ', ' + t[0]
    elif t[1] == 'NORP':
      ner[NORP] = ner[NORP] + ', ' + t[0]
    elif t[1] == 'FAC':
      ner[FAC] = ner[FAC] + ', ' + t[0]
    elif t[1] == 'ORG':
      ner[ORG] = ner[ORG] + ', ' + t[0]
    elif t[1] == 'GPE':
      ner[GPE] = ner[GPE] + ', ' + t[0]
    elif t[1] == 'LOC':
      ner[LOC] = ner[LOC] + ', ' + t[0]
    elif t[1] == 'PRODUCT':
      ner[PRODUCT] = ner[PRODUCT] + ', ' + t[0]
    elif t[1] == 'EVENT':
      ner[EVENT] = ner[EVENT] + ', ' + t[0]
    elif t[1] == 'WORK_OF_ART':
      ner[WORK_OF_ART] = ner[WORK_OF_ART] + ',' + t[0]
    elif t[1] == 'LAW':
      ner[LAW] = ner[LAW] + ', ' + t[0]
    elif t[1] == 'DATE':
      ner[DATE] = ner[DATE] + ', ' + t[0]
    elif t[1] == 'TIME':
      ner[TIME] = ner[TIME] + ', ' + t[0]
    elif t[1] == 'MONEY':
      ner[MONEY] = ner[MONEY] + ', ' + t[0]
  
  # strip first comma ','
  for i in range(0, len(ner)):
    if len(ner[i]) > 2:
      ner[i] = ner[i][2:]

  return ner

def getDocNer(content):
  # Spacy magic
  doc = nlp(content)

  # Ner (Named Entity Recognition)
  ner = getNer(doc)

  return doc, ner