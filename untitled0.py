# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 11:35:48 2022

@author: Carlos de Cloet
"""

import re
import json
import time
import random
from sklearn.metrics import jaccard_score
from random import shuffle
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import bootstrap 

starttime = time.time()

# Loading data
f = open('TVs-all-merged.json', 'r')
data = json.load(f)

dataList = []

for i in data:
    dataList.append(data[i])

tvs = []

for i in range(len(dataList)):
    for j in range(len(dataList[i])):
        tvs.append(dataList[i][j])

titles = []
for dct in tvs:
        dct['title'] = dct['title'].lower()
        dct['title'] = dct['title'].replace("inches","inch")
        dct['title'] = dct['title'].replace("-inch","inch")
        dct['title'] = dct['title'].replace(" inch","inch")
        dct['title'] = dct['title'].replace("\"","inch")
        dct['title'] = dct['title'].replace("("," ")
        dct['title'] = dct['title'].replace(")"," ")
        
        dct['title'] = dct['title'].replace("hertz","hz")
        dct['title'] = dct['title'].replace(" hertz","hz")
        dct['title'] = dct['title'].replace("-hz","hz")
        dct['title'] = dct['title'].replace(" hz","hz")
        
        titles.append(dct['title'])
        
        
wordsWithStringAndChar = set()
for i in titles:
    modelwords = re.finditer('[a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*' ,i)
    for candidateWord in modelwords:
        wordsWithStringAndChar.add(candidateWord.group())
        
        

newdata = random.choices(tvs, k = int(len(tvs)))
   

new_list = []

for dictionary in newdata:
    if dictionary not in new_list:
        new_list.append(dictionary)
        
testData = []

for dictionary in tvs:
    if dictionary not in new_list:
        testData.append(dictionary)