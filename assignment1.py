import numpy as np
import pandas as pd
import numbers
import decimal
import math
import logisticRegression as lR
import linearDiscriminantAnalysis as lDA
import scipy.stats as ss
import matplotlib.pyplot as plt
from statistics import stdev
from statistics import mean
import time
 
class Stats:
 
    @staticmethod
    def removeOutliers(data):
        rowsToDelete = []
        for a in range(len(data.iloc[1,:])):
            feature = data.iloc[:,a]
            avg = mean(feature)
            stdv = stdev(feature)
            for b in range(0,len(feature)):
                if feature.iloc[b] < avg - 3.5*stdv or feature.iloc[b] > avg + 3.5*stdv:
                    if not rowsToDelete.__contains__(b):
                        rowsToDelete.append(b)
        
        print("ROWS DELETED: ")
        print(len(rowsToDelete))
 
        data.drop(rowsToDelete, inplace = True, axis = 0)
        return data
    
    @staticmethod
    def normalityOfFeatures(data):
        alphas = []
        for a in range(len(data.iloc[1,:])):
            feature = data.iloc[:,a]
            k2,p = ss.normaltest(feature)
            alphas.append(p)
        print(alphas)
        return alphas
 
    @staticmethod
    def ratioOfOnes(output):
        ones = 0
        for a in output:
            if a == 1:
                ones = ones + 1
        ratio = ones / len(output) * 100
        return ratio
 
#auxiliary functions
 
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
 
print("-----------TASK 1---------------")
 
#Replace this with just the name of file if on PC
wine_data = pd.read_csv('winequality-red.csv', sep=';',header=None)
cancer_data = pd.read_csv('breast-cancer-wisconsin.data', sep=',',header=None)
 
def deleteMalformedRows(d):
    rowsToDelete = []
    co = 0
    for index, row in d.iterrows():
        for cell in row:
            if not is_number(cell):
                rowsToDelete.append(co)
                break
        co = co + 1
    
    d.drop(rowsToDelete, inplace = True)
    return d
 
wine_data = deleteMalformedRows(wine_data)
cancer_data = deleteMalformedRows(cancer_data)
 
def convertLastColumnToBinary(d):
    c = 0 #counter
    for num in d.iloc[:,-1]: #for each value in last column
        if (float(num) < 6.0):
            d.iat[c,-1] = 0 #update array
        else:
            d.iat[c,-1] = 1
        c = c + 1
    return d
 
def convertCancerToBinary(d):
    c = 0 #counter
    for num in d.iloc[:,-1]: #for each value in last column
        if (float(num) == 2):
            d.iat[c,-1] = 0 #update array
        else:
            d.iat[c,-1] = 1
        c = c + 1
    return d
 
wine_data = convertLastColumnToBinary(wine_data)
cancer_data = convertCancerToBinary(cancer_data)
 
def convertToNum(a):
    for x in range(0, a.shape[0]):
        for y in range(0, a.shape[1]):
            a.iloc[x,y] = float(a.iloc[x,y])
 
 
 
convertToNum(wine_data)
convertToNum(cancer_data)
 
#wine_data = Stats.removeOutliers(wine_data)
#Stats.normalityOfFeatures(wine_data)
#ratioOnes = Stats.ratioOfOnes(wine_data.iloc[:,-1])
print("-------logistic regression--------------")
"""
wineLR = lR.logisticRegression(wine_data.iloc[:,:-1], wine_data.iloc[:,-1], 0.1, 100, 0,5) 
start = time.time()
acc=wineLR.start()
end = time.time()
print("wine")
print(acc)
print(end-start)

cancerLR = lR.logisticRegression(cancer_data.iloc[:,:-1], cancer_data.iloc[:,-1], 0.1, 100, 0,5) 
start = time.time()
acc = cancerLR.start()
end = time.time()
print("cancer")
print(acc)
print(end-start)
"""
#print("-----------------LDA------------------------------") 
print("---------linear discriminant analysis------------")
var = lDA.linearDiscriminantAnalysis(cancer_data.iloc[:,:-1])
var = lDA.linearDiscriminantAnalysis(wine_data.iloc[:,:-1])
wd=wine_data.iloc[:,0:11]
x0=wine_data.iloc[3:4,0:11]
#(ans,inc,cor)=var.predict_A(wd,wine_data.iloc[:,-1])
start = time.time()
myl=var.predict_k(wd,wine_data.iloc[:,-1],5)
end = time.time()
print("wine")
print(*myl, sep = ", ")
print(end-start)
print(sum(myl)/len(myl))
start = time.time()
myl=var.predict_k(cancer_data.iloc[:,0:10],cancer_data.iloc[:,-1],5)
end = time.time()
print("cancer")
print(*myl, sep = ", ") 
print(end-start)
print(sum(myl)/len(myl))
print("---------LDA k-fold tests------------")
"""k_lda=[]
k_lda_avg=[]
k_lda_time=[]
y=math.log2(np.shape(wd)[0])
x=int(y)
for a in range(1,x,1):
    b=a
    b=int(math.ldexp(1,a))
    print("starting %d" %(b))
    start = time.time()
    myl=var.predict_k_QDA(wd,wine_data.iloc[:,-1],b)
    end = time.time()
    k_lda.append(b)
    k_lda_avg.append(sum(myl)/len(myl))
    k_lda_time.append(end-start)

print("starting last one")
start = time.time()
myl=var.predict_k_QDA(wd,wine_data.iloc[:,-1],(np.shape(wd)[0]))
end = time.time()
k_lda.append((np.shape(wd)[0]))
k_lda_avg.append(sum(myl)/len(myl))
k_lda_time.append(end-start)
print(*k_lda, sep = ", ") 
print(*k_lda_avg, sep = ", ") 
print(*k_lda_time, sep = ", ") 
"""
"""
print("---------LDA k-fold tests cancer------------")
k_lda=[]
k_lda_avg=[]
k_lda_time=[]
y=math.log2(np.shape(cancer_data.iloc[:,0:10])[0])
x=int(y)
for a in range(1,x,1):
    b=a
    b=int(math.ldexp(1,a))
    print("starting %d" %(b))
    start = time.time()
    myl=var.predict_k(cancer_data.iloc[:,0:10],cancer_data.iloc[:,-1],b)
    end = time.time()
    k_lda.append(b)
    k_lda_avg.append(sum(myl)/len(myl))
    k_lda_time.append(end-start)
print(*k_lda, sep = ", ") 
print(*k_lda_avg, sep = ", ") 
print(*k_lda_time, sep = ", ") 
"""
"""print("starting last one")
start = time.time()
myl=var.predict_k_Log_odds(cancer_data.iloc[:,0:10],cancer_data.iloc[:,-1],(np.shape(cancer_data.iloc[:,0:10])[0]))
end = time.time()
k_lda.append((np.shape(cancer_data.iloc[:,0:10])[0]))
k_lda_avg.append(sum(myl)/len(myl))
k_lda_time.append(end-start)
print(*k_lda, sep = ", ") 
print(*k_lda_avg, sep = ", ") 
print(*k_lda_time, sep = ", ") 
""""""
myl=var.predict_k(wd,wine_data.iloc[:,-1],5)

print("LDA accuracy wine Discriminant")
print(*myl, sep = ", ") 
print(sum(myl)/len(myl))
myl=var.predict_k_Log_odds(wd,wine_data.iloc[:,-1],5)
print("LDA accuracy wine Log odds")
print(*myl, sep = ", ") 
print(sum(myl)/len(myl))
myl=var.predict_k_QDA(wd,wine_data.iloc[:,-1],5)

print("QDA accuracy wine")
print(*myl, sep = ", ") 
print(sum(myl)/len(myl))


myl=var.predict_k(cancer_data.iloc[:,0:10],cancer_data.iloc[:,-1],5)
print("LDA accuracy cancer Discriminant")
print(*myl, sep = ", ") 
print(sum(myl)/len(myl))
myl=var.predict_k_Log_odds(cancer_data.iloc[:,0:10],cancer_data.iloc[:,-1],5)
print("LDA accuracy cancer Log odds")
print(*myl, sep = ", ") 
print(sum(myl)/len(myl))
myl=var.predict_k_QDA(cancer_data.iloc[:,0:10],cancer_data.iloc[:,-1],5)
print("QDA accuracy cancer")
print(*myl, sep = ", ") 
print(sum(myl)/len(myl)) 

" 
 
print("---------TASK 3----------")
"""
"""
print("1")
learningRates = [0.01, 0.1, 1, 10, 100]
for b in learningRates:
    cancerLR = lR.logisticRegression(cancer_data.iloc[:,:-1], cancer_data.iloc[:,-1], b, 10, 0,5) 
    acc = cancerLR.start()
    print(acc)
 
print("2")
 
t = time.time()
 
cancerLR = lR.logisticRegression(cancer_data.iloc[:,:-1], cancer_data.iloc[:,-1], 0.1, 10, 0,5) 
cancerLRaccuracy = cancerLR.start()
 
t2 = time.time()
 
cancerTime = t2 - t
 
start = time.time()
 
wineLR = lR.logisticRegression(wine_data.iloc[:,:-1], wine_data.iloc[:,-1], 0.1, 1000, 0,5)
wineLRaccuracy = wineLR.start() 
 
end = time.time()
 
wineTime = end - start
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
#Removing malformed features
 
#delete rows for wine
 
 
 
#rowsToDeleteWine = []
#co = 0
 
#for index, row in wine_data.iterrows():
  #  for cell in row:
   #     if not is_number(cell):
    #        rowsToDeleteWine.append(co)
    #        break
   # co = co + 1
 
 
#for r in rowsToDeleteWine:
  #  wine_data.drop([r], inplace = True)
 
#delete rows for cancer
 
#rowsToDeleteCancer = []
#cou = 0
#
#for index, row in cancer_data.iterrows():
   # for cell in row:
      #  if not is_number(cell):
         #   rowsToDeleteCancer.append(cou)
         #   break
    #cou = cou + 1
 
#for r in rowsToDeleteCancer:
   # cancer_data.drop([r], inplace = True)
 
#Convert wine_data to binary
#guaranteed to include only numbers at this point
#c = 0 #counter
#for num in wine_data.iloc[:,-1]: #for each value in last column
    #if (float(num) < 6.0):
     #   wine_data.iat[c,-1] = 0 #update array
    #else:
      #  wine_data.iat[c,-1] = 1
    #c = c + 1
 
 
 
 
 
 
 
 
 
 
 
 
 
"""
 
