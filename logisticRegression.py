import math
import numpy as np
from statistics import mean
 
class logisticRegression:
 
    def __init__(self, input, output, learningRate, descents, initialWeight, folds):
        self.input = input
        self.output = output
        self.learningRate = learningRate
        self.descents = descents
        self.initialWeight = initialWeight
        self.folds = folds
 
    def sigma(self,x):
        return 1 / (1 + np.exp((-(x))))
 
    def updateFunction(self,x,y,w):
        w = np.array(w) #for syntax purposes
        
        cur = [0.0] * len(w) #holder array that will keep track of the sum 
        
        for z in range(0,len(x.iloc[:,1])): #for each row
            l = np.multiply(x.iloc[z,:],(np.subtract(y.iloc[z],self.sigma(np.dot(w.T,x.iloc[z,:]))))) #update function
            cur = np.add(cur,l)
        return w + np.multiply(self.learningRate,cur) #updated w
 
    def predict(self, x, w):
        sum = 0
        for z in range(len(w)):
            sum = sum + w[z] * x.iloc[z]
        return self.sigma(sum)
    
    def findW(self, inp, outp):
        w = [self.initialWeight] * len(inp.iloc[1,:]) #initialize w array with initial weight
        for c in range(0,self.descents): #do update rule on x descents
            w = self.updateFunction(inp, outp, w)
        return w #final w
 
    def crossFoldExamination(self):
        rows = len(self.input.iloc[:, 1]) #number rows
        foldLength = int(rows / self.folds) #rows per fold
 
        ws = [] #all of the final w arrays for each fold
        tests = [] #all of the test data that was held out
        accuracies = []
        for a in range(0,self.folds*foldLength, foldLength): #foreach fold
            training = self.input.drop(self.input.index[a:a + foldLength]) #training array    
            resTraining = self.output.drop(self.output.index[a:a + foldLength]) #actual results
            testInp = self.input.iloc[a: a + foldLength] #fold for testing
            testOutp = self.output.iloc[a: a + foldLength] #result of fold for testing
 
            w = self.findW(training, resTraining)
            accuracy = self.findAccuracy(w, testInp, testOutp)
 
            ws.append(w)
            accuracies.append(accuracy)
        return [ws, accuracies]
 
    #finds best accuracy given all of the accuracies
    def findBestAcc(self, ws, accuracies):
        maxi = 0
        bestTuple = []
        for a in range(len(accuracies)):
            if accuracies[a] > maxi:
                maxi = accuracies[a]
                bestTuple = (ws[a], accuracies[a])
        return bestTuple
 
    #determines binary classificatoin given threshold
    def threshold(self, num):
        if (num >= 0.5):
            return 1
        else:
            return 0
 
    def findAccuracy(self, w, inp, outp):
        correct = 0.0
        false_positive=0
        false_negative=0
        for b in range(0,len(inp.iloc[:,1])):
            prediction = self.predict(inp.iloc[b,:], w)
            result = self.threshold(prediction)
            
            if result == outp.iloc[b]:
                correct = correct + 1.0
            elif result==1:
                false_positive+=1
            else:
                false_negative+=1
        if (len(inp.iloc[:,1])==len(self.input.iloc[:,1])):
                print("false positive")
                print(false_positive)
                print("false negative")
                print(false_negative)
                print("total")
                print(len(inp.iloc[:,1]))    
            
        return correct / (len(inp.iloc[:,1])) * 100.0
 
    """
    def addFeatures(self):
        x = self.input
        newColumns = []
        for a in range(len(self.input.iloc[1,:])):
            b = []
            for c in range(len(self.input.iloc[:,1])):
                val = self.input.iloc[c,a]
                b.append(val + val * val)
            newColumns.append(b)
        self.input.append(newColumns, axis = 1)
    """
 
    def start(self):
        ws, accuracies = self.crossFoldExamination()
 
        print("ACCURACIES")
        print(accuracies)
        bestCombo = self.findBestAcc(ws, accuracies)
        bestW = bestCombo[0]
        bestAcc = bestCombo[1]
 
        print("BEST W")
        print(bestW)
 
        print("BEST ACCURACY")
        print(bestAcc)
 
        averageWeight = [0]*len(bestW)
        for b in ws:
            averageWeight = np.add(averageWeight,b)
        averageWeight = averageWeight / len(ws)
 
        print("AVERAGE WEIGHTS")
        print(averageWeight)
 
        print("MEAN ACCURACY")
        print(mean(accuracies))
 
        print("Average weight tested on entire input")
        print(self.findAccuracy(bestW, self.input, self.output))
        return mean(accuracies)
 
 
 
 
 
 
 
        
 
    
 
    
 
        
 
 
 
 
