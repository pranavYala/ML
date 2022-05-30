import numpy as np
import pandas as pd
import re
import string
import math
import nltk
import gensim
import random
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim import models

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

#pre-process list of skills

ex_data = pd.read_csv('Example_Technical_Skills.csv', usecols=['Technology Skills'])

for i in range(len(ex_data['Technology Skills'])):
    ex_data['Technology Skills'][i] = ex_data['Technology Skills'][i].lower()

ex_data['Technology Skills'] = ex_data['Technology Skills'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ' , x))

stop_words = set(stopwords.words('english'))
stop_words.add('subject')
stop_words.add('http')

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

ex_data['Technology Skills'] = ex_data['Technology Skills'].apply(lambda x: remove_stopwords(x))

lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

ex_data["Technology Skills"] = ex_data["Technology Skills"].apply(lambda text: lemmatize_words(text))

ex_data["Technology Skills"] = ex_data["Technology Skills"].apply(lambda x: re.sub(' +', ' ', x))

for i in range(len(ex_data['Technology Skills'])):
    ex_data['Technology Skills'][i] = ex_data['Technology Skills'][i].split()
    
#pre-process list of raw_data

data = pd.read_csv('Raw_Skills_Dataset.csv', usecols=['RAW DATA'])

for i in range(len(data['RAW DATA'])):
    data['RAW DATA'][i] = data['RAW DATA'][i].lower()

data = data.drop_duplicates(subset='RAW DATA')
or_data = data

data['RAW DATA'] = data['RAW DATA'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ' , x))

data['RAW DATA'] = data['RAW DATA'].apply(lambda x: remove_stopwords(x))

data["RAW DATA"] = data["RAW DATA"].apply(lambda text: lemmatize_words(text))

data["RAW DATA"] = data["RAW DATA"].apply(lambda x: re.sub(' +', ' ', x))

for i, r in data.iterrows():
    r['RAW DATA'] = r['RAW DATA'].split()

#training the model

trainer = []
for i in range(len(ex_data['Technology Skills'])):
    trainer.append(ex_data['Technology Skills'][i])
    
model = Word2Vec(sentences=trainer, sg=1, vector_size=100, workers=4)

ans = []
for i, r in data.iterrows():
    sum = 0
    
    for j in range(len(r['RAW DATA'])):
        for k in range(len(ex_data['Technology Skills'])):
            for x in range(len(ex_data['Technology Skills'][k])):
                try:
                    sum -= model.wv.distance(r['RAW DATA'][j], ex_data['Technology Skills'][k][x])
                except:
                    continue
        
    if (sum < 0 and sum > -1500):
        ans.append(or_data.loc[[i]]['RAW DATA'])
                        
print(ans)
