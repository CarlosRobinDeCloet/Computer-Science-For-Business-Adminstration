# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:17:18 2022

@author: Carlos de Cloet
"""


import re
import json
import time
import random
import numpy as np
from scipy.cluster import hierarchy
from numpy.linalg import norm
from scipy.cluster.hierarchy import dendrogram
starttime = time.time()

# Loading data
f = open('TVs-all-merged.json', 'r')
data = json.load(f)


listWithBrands = ['samsung', 'vizio', 'westinghouse', 'jvc tv', 'sony', 'philips', 'toshiba', 'sharp', 'upstar', 'jvc', 'sceptre', 'sanyo', 'sunbritetv', 'proscan', 'lg electronics', 'seiki', 'pansonic', 'lg', 'craig', 'panasonic', 'sansui', 'viewsonic', 'hannspree', 'toshiba', 'hisense', 'supersonic', 'rca', 'coby', 'haier', 'sigmac']

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
        
        dct['brand'] = []
        search_brand = dct['title'].split()
        for word in search_brand:
            if word in listWithBrands:
                dct['brand'] = word
        
        if dct['brand'] == 'jvc tv':
            dct['brand'] = 'jvc'
            
        if dct['brand'] == 'lg electronics':
            dct['brand'] = 'lg'
        
def hashFunction(a,b,p,x):
    return (a+b*(x))%p

def jaccard_similarity(x,y):
    intersection = x.intersection(y)
    union = x.union(y)
    similarity = len(intersection) / float(len(union))
    return similarity

def cosine_similarity(x,y):
    dotprod = np.dot(x,y)
    normX = norm(x)
    normY = norm(y)
    similarity = dotprod/(normX*normY)
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

def binVectors(rowPart: list, bucket: list):

    for i in range(len(rowPart)):
        for j in rowPart[i]:
            stringOfNumber = str()
            for k in range(len(j)):
                stringOfNumber += str(j[k])
            number = int(stringOfNumber)
            number = number%9999991
            buckets[number].append(i)

def bootstrapping(tvs: list):
    bootstrappedData = random.choices(tvs, k = int(len(tvs)))
    
    trainingData = []
    for dictionary in bootstrappedData:
        if dictionary not in trainingData:
            trainingData.append(dictionary)
            
    testData = []
    for dictionary in tvs:
        if dictionary not in trainingData:
            testData.append(dictionary)
    return trainingData, testData  

def classifyAsDuplicates(clf_matrix,dis_matrix, threshold: float):
    for i in range(len(clf_matrix)):
        for j in range(len(clf_matrix)):
            if i < j:
                if dis_matrix[i][j] < threshold:
                    clf_matrix[i][j] = True
                    
   
def calculatingPrecisionAndRecall(clf_matrix, tvs):
    
    tp = 0
    fp = 0
    fn = 0
    count = 0
    
    for i in range(len(clf_matrix)):
        for j in range(len(clf_matrix)):
            if i < j:
                if clf_matrix[i][j] == 1:
                    if tvs[i]['modelID'] == tvs[j]['modelID']:
                        tp += 1
                        count += 1
                    else:
                        fp += 1
                        count += 1
                if clf_matrix[i][j] == 0:
                    if tvs[i]['modelID'] == tvs[j]['modelID']:
                        fn += 1
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    
    print("")
    print("Count is: " + str(count))
    print("")
    print("tp = " + str(tp))
    print("fp = " + str(fp))
    print("fn = " + str(fn))
    print("")
    print("Precision = " + str(precision))
    print("Recall = " + str(recall))
    print("")
    f1 = 2*precision*recall/(precision+recall)
    return f1, precision, recall                    
   
PQBS = []    
PCBS = []
F1asteriskBS = []
F1BS  = []   
PR = []
RC = []

results = []
bands = [65] #[5, 10, 13, 25, 26, 50, 65, 130, 325]
    
    #####################################################################################################################
    # Bootstrapped data.
    #####################################################################################################################
for band in range(len(bands)):
    results.append([])
    for i in range(5):                                                                
        
        print("Iteration " + str(i+1) + " of bootstrapping the data. Elapsed time is: " + str(time.time()-starttime))

        bootstrappedData = bootstrapping(tvs)
        
        trainingData = bootstrappedData[0]
        testData = bootstrappedData[1]
        n = len(trainingData)
        
        titlesTrain = []
        for dict in trainingData:
            titlesTrain.append(dict['title'])
            dict['modelwords'] = []
       
        titlesTest = []
        for dict in testData:
            titlesTest.append(dict['title'])
            dict['modelwords'] = []
    
        wordsWithStringAndChar = set()
        countWordsWithStringAndChar = []
        for i in range(len(titlesTrain)):
            modelwords = re.finditer('[a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*' ,titlesTrain[i])
            for candidateWord in modelwords:
                trainingData[i]['modelwords'].append(candidateWord.group())
                wordsWithStringAndChar.add(candidateWord.group())
                countWordsWithStringAndChar.append(candidateWord.group())
            
        for products in trainingData:
            for key in products['featuresMap'].keys():
                modelwords = re.finditer('(\d+\:+\d)',products['featuresMap'][key])
                for candidateWord in modelwords:
                    trainingData[i]['modelwords'].append(candidateWord.group())
                    wordsWithStringAndChar.add(candidateWord.group())
                    countWordsWithStringAndChar.append(candidateWord.group())

        listOfWords = list(wordsWithStringAndChar)
   
        oneHotEncoded = []

        for i in range(len(titlesTrain)):
            oneHotEncoded.append([1 if x in titlesTrain[i].split() else 0 for x in listOfWords])
    
        signatureMatrix = np.empty([650,len(oneHotEncoded)]) # 650 equals 50% of the model words
        signatureMatrix.fill(99999)

        a = np.random.randint(0,1000,size = len(signatureMatrix))
        b = np.random.randint(1,1000,size = len(signatureMatrix))
        prime = 4549
        
        for r in range(len(oneHotEncoded[0])):
     
            hashValues = hashFunction(a,b,prime,r)
        
            #if r%50 == 0:
            #    print('Current iteration for minhashing is: ' + str(r) + " Elapsed time is: " + str(time.time()-starttime))
    
    
            for i in range(len(oneHotEncoded)):
                if oneHotEncoded[i][r] == 1:
                    for j in range(len(signatureMatrix)):
                        if hashValues[j] < signatureMatrix[j][i]:
                            signatureMatrix[j][i] = hashValues[j]
            
        for i in range(len(signatureMatrix)):
            for j in range(len(signatureMatrix[i])):
                if signatureMatrix[i][j] == 99999:
                    print('ohno')
    
                                                                      
        ## 650 is divisible by {1, 2, 5, 10, 13, 15, 25, 26, 50, 65, 130, 325, and 650}

        subVectorMatrix = split_signatureMatrix(signatureMatrix, bands[band])
        print("Finished splitting vectors. Elapsed time is: " + str(time.time() - starttime))

        buckets = []
        for i in range(10000000):
            buckets.append([])  
    
        binVectors(subVectorMatrix, buckets)
        print("Finished bucketing vectors. Elapsed time is: " + str(time.time() - starttime))

        listOfCandidatePairs = []
        numCol = 0
        for i in range(len(buckets)):
            if len(buckets[i]) > 1:
                listOfCandidatePairs.append([])
                for j in range(len(buckets[i])):
                    listOfCandidatePairs[numCol].append(buckets[i][j])
                numCol += 1
                
        print("Finished gathering candidate pairs. Elapsed time is: " + str(time.time() - starttime))  

        candidatePairsMatrix = np.empty([n,n])
        candidatePairsMatrix.fill(False)

        for i in listOfCandidatePairs:
            for j in i:
                for k in i:
                    candidatePairsMatrix[j][k] = True
                    if trainingData[j]['shop'] == trainingData[k]['shop']:
                        candidatePairsMatrix[j][k] = False
                    if trainingData[j]['brand'] != [] and trainingData[k]['brand'] != [] and trainingData[j]['brand'] != trainingData[k]['brand']:
                        candidatePairsMatrix[j][k] = False
         
        duplicateFound = 0
        duplicateTotal = 0
        amountOfComparisonsMade = 0
        for i in range(len(candidatePairsMatrix)):
            for j in range(len(candidatePairsMatrix[i])):
                if i < j:
                    if trainingData[i]['modelID'] == trainingData[j]['modelID']:
                        duplicateTotal += 1
                    if candidatePairsMatrix[i][j] == 1:
                        amountOfComparisonsMade += 1
                        if trainingData[i]['modelID'] == trainingData[j]['modelID']:
                            duplicateFound += 1
                           
        pairQuality = duplicateFound/amountOfComparisonsMade
        pairCompleteness = duplicateFound / duplicateTotal
        F1 = (2*pairQuality*pairCompleteness)/(pairQuality+pairCompleteness)

        #USING COSINE SIMILARITY
        dissimilarityMatrix = np.empty([n,n])
        dissimilarityMatrix.fill(99999)
    

        for i in range(len(candidatePairsMatrix)):
            for j in range(len(candidatePairsMatrix[i])):
                if i < j:
                    if candidatePairsMatrix[i][j] == 1:
                        #modelWordsI = set(trainingData[i]['modelwords'])
                        #modelWordsJ = set(trainingData[j]['modelwords'])
                        #dissimilarityMatrix[i,j] = 1 - jaccard_similarity(modelWordsI, modelWordsJ)
                                           
                        binaryVectorI = oneHotEncoded[i]
                        binaryVectorJ = oneHotEncoded[j]
                        dissimilarityMatrix[i,j] = 1 - cosine_similarity(binaryVectorI,binaryVectorJ)
                    
        predictedDuplicateMatrix = np.empty([n,n])
        predictedDuplicateMatrix.fill(False)

        # Threshold should be between 0 and 1    
        classifyAsDuplicates(predictedDuplicateMatrix, dissimilarityMatrix, 0.2)    
    
        print('Calculating similarities finished. Elapsed time is: ' + str(time.time() - starttime))       
 
        f1 = calculatingPrecisionAndRecall(predictedDuplicateMatrix, trainingData)

        endtime = time.time()
    
        PCBS.append(pairCompleteness)
        PQBS.append(pairQuality)
        F1BS.append(f1[0])
        F1asteriskBS.append(F1)
        PR.append(f1[1])
        RC.append(f1[2])

        print("")
        print("The pair quality is: " + str(pairQuality*100) + "%.")
        print("The pair completeness is: " + str(pairCompleteness*100) + "%.")
        print("The F1* score is: " + str(F1))
        print("The F1 score is: " + str(f1[0]))
        print("")
    
    print("Program finished. Total elapsed time is: " + str(endtime - starttime))

    avgPQ = np.average(PQBS)
    avgPC = np.average(PCBS)
    avgF1asterisks = np.average(F1asteriskBS)
    avgF1 = np.average(F1BS)
    avgPR = np.average(PR)
    avgRC = np.average(RC)
        
    print("")
    print("The average pair quality is: " + str(avgPQ*100) + "%.")
    print("The average pair completeness is: " + str(avgPC*100) + "%.")
    print("The average F1* score is: " + str(avgF1asterisks))
    print("The average F1 score is: " + str(avgF1))
    print("The average precision score is: " + str(avgPR))
    print("The average recall score is: " + str(avgRC))
    print("")
    
    results[band].append(avgPQ)
    results[band].append(avgPC)
    results[band].append(avgF1asterisks)
    results[band].append(avgF1)
    results[band].append(avgPR)
    results[band].append(avgRC)