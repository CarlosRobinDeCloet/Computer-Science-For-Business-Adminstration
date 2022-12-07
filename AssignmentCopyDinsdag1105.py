# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:17:18 2022

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


listOfWords = list(wordsWithStringAndChar)

oneHotEncoded = []

for i in range(len(titles)):
    oneHotEncoded.append([1 if x in titles[i].split() else 0 for x in listOfWords])

def hashFunction(a,b,p,x):
    return (a+b*(x))%p

prime = 4549

signatureMatrix = np.empty([650,len(oneHotEncoded)]) ### [oneHotEncoded/2, oneHotEncoded]
signatureMatrix.fill(99999)


a = np.random.randint(0,1000,size = len(signatureMatrix))
b = np.random.randint(1,1000,size = len(signatureMatrix))

for r in range(len(oneHotEncoded[0])):
     
    hashValues = hashFunction(a,b,prime,r)
    print('Current iteration for minhashing is: ' + str(r) + " Elapsed time is: " + str(time.time()-starttime))
    
    
    for i in range(len(oneHotEncoded)):
        #minHashValue = signatureMatrix[r][i]
        if oneHotEncoded[i][r] == 1:
            for j in range(len(signatureMatrix)):
                if hashValues[j] < signatureMatrix[j][i]:
                    signatureMatrix[j][i] = hashValues[j]
            
for i in range(len(signatureMatrix)):
    for j in range(len(signatureMatrix[i])):
        if signatureMatrix[i][j] == 99999:
            print('ohno')
    
                                                                      
# Jaccard Similarity
def jaccard_similarity(x,y):
    """A function for finding the similarity between two sets"""
    "Source: https://www.learndatasci.com/glossary/jaccard-similarity/"
    intersection = x.intersection(y)
    union = x.union(y)
    similarity = len(intersection) / float(len(union))
    return similarity


def split_signatureMatrix(signature, b: int):
    assert len(signature) % b == 0
    r = int(len(signature)/b)

    splittedColumns = []
    
    for c in range(len(signature[0])):
        splittedColumns.append([])
        colNum = 0
        for i in range(0, len(signature), r):
            splittedColumns[c].append([])
            for j in range(i,i+r):
                splittedColumns[c][colNum].append(int(signature[j][c]))
            colNum = colNum + 1    
    return splittedColumns

## 650 is divisible by {1, 2, 5, 10, 13, 15, 25, 26, 50, 65, 130, 325, and 650}

subVectorMatrix = split_signatureMatrix(signatureMatrix, 50)
print("Finished splitting vectors. Elapsed time is: " + str(time.time() - starttime))


buckets = []
for i in range(10000000):
    buckets.append([])  
    
def binVectors(rowPart: list, bucket: list):

    for i in range(len(rowPart)):
        for j in rowPart[i]:
            stringOfNumber = str()
            for k in range(len(j)):
                stringOfNumber += str(j[k])
            number = int(stringOfNumber)
            number = number%9999991
            buckets[number].append(i)
        
binVectors(subVectorMatrix, buckets)
print("Finished bucketing vectors. Elapsed time is: " + str(time.time() - starttime))


dissimilarityMatrix = np.empty([1624,1624])
dissimilarityMatrix.fill(99999)

## USING JACCARD SIMILARITY AS FIRST EXPLORATION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#for i in range(len(buckets)):    
#    if len(buckets[i]) > 0:
#        for j in buckets[i]:
#            words_doc1 = set(titles[j].split())
#            for k in buckets[i]:
#                words_doc2 = set(titles[k].split())
#                dissimilarityMatrix[j,k] = 1 - jaccard_similarity(words_doc1, words_doc2)
                
#print('Calculating similarities finished. Elapsed time is: ' + str(time.time() - starttime))       
 
listOfCandidatePairs = []
numCol = 0
for i in range(len(buckets)):
    if len(buckets[i]) > 1:
        listOfCandidatePairs.append([])
        for j in range(len(buckets[i])):
            listOfCandidatePairs[numCol].append(buckets[i][j])
        numCol += 1
print("Finished gathering candidate pairs. Elapsed time is: " + str(time.time() - starttime))  

duplicateTotal = 0
for i in data.keys():
    
    sumDuplicateInKey = 0
    amountOfItemsInKey = 0
    for j in data[i]:
        amountOfItemsInKey += 1
    
    if amountOfItemsInKey > 1:
        sumDuplicateInKey += amountOfItemsInKey -1
      
    duplicateTotal += sumDuplicateInKey

candidatePairsMatrix = np.empty([1624,1624])
candidatePairsMatrix.fill(False)

for i in listOfCandidatePairs:
    for j in i:
        for k in i:
            candidatePairsMatrix[j][k] = True

                    
duplicateFound = 0
amountOfComparisonsMade = 0
for i in range(len(candidatePairsMatrix)):
    for j in range(len(candidatePairsMatrix[i])):
        if i < j:
            if candidatePairsMatrix[i][j] == 1:
                amountOfComparisonsMade += 1
                if tvs[i]['modelID'] == tvs[j]['modelID']:
                    duplicateFound += 1
                
            
pairQuality = duplicateFound/amountOfComparisonsMade
print(pairQuality)

pairCompleteness = duplicateFound / duplicateTotal
print(pairCompleteness)

F1 = (2*pairQuality*pairCompleteness)/(pairQuality+pairCompleteness)

            
#clustering = linkage(dissimilarityMatrix)   
#dn = dendrogram(clustering)
            
endtime = time.time()

print("")
print("The amount of bands is: " + str(13) + "and the amount of rows is: " + str(650/13))
print("The pair quality is: " + str(pairQuality*100) + "%.")
print("The pair completeness is: " + str(pairCompleteness*100) + "%.")
print("The F1 score is: " + str(F1*100))
print("")
print("Program finished. Total elapsed time is: " + str(endtime - starttime))



