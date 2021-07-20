import math
import numpy as np
import pandas as pd
import sys

class linearDiscriminantAnalysis:
    def __init__(self, input): 
    # output):
        self.input = input
        # self.output = output

    
    def NumberOfPositiveValues(self, x):
        num1 = np.count_nonzero(x.iloc[:,-1])
        total = x.shape[0]
        # row_count = sum(1 for row in x) 
        probOf1 = num1 / total
        print(probOf1)
        
    def NumberOfNegativeValues(self, x):
        num1 = x.shape[0] - np.count_nonzero(x.iloc[:,-1])
        total = x.shape[0]  
        # row_count = sum(1 for row in x) 
        probOf0 = num1 / total
        print(probOf0) 

    
    
    def predict_A (self,X,y,CovI,u0,u1,P0,P1):
        
        #(CovI,u0,u1,P0,P1)=self.fit(X,y)
        correct=0.0;
        incorrect=0.0;
        false_pos=0
        false_neg=0
        (a,b)=np.shape(X)
        for i in range(np.shape(X)[0]):
            #goes through ever
            x0=X.iloc[i:i+1,0:b]
            ans=self.predict(CovI,u0,u1,P0,P1,x0)
            if ans==y.iloc[i]:
                correct+=1
            else:
                incorrect+=1
                if ans==1:
                    false_pos+=1
                else:
                    false_neg+=1
        print("false pos")
        print(false_pos)
        print("false neg")
        print(false_neg)
        print("false pos")
        print(incorrect+correct)
        return (correct/(incorrect+correct))
    def predict_A_Log_odds (self,X,y,CovI,u0,u1,P0,P1):
        
        #(CovI,u0,u1,P0,P1)=self.fit(X,y)
        correct=0.0;
        incorrect=0.0;
        (a,b)=np.shape(X)
        for i in range(np.shape(X)[0]):
            x0=X.iloc[i:i+1,0:b]
            ans=self.predict_Log_odds(CovI,u0,u1,P0,P1,x0)
            if ans==y.iloc[i]:
                correct+=1
            else:
                incorrect+=1
        return (correct/(incorrect+correct))
    def predict_A_QDA (self,X,y,CovI0,CovI1,det0,det1,u0,u1,P0,P1):
        
        #(CovI0,CovI1,det0,det1,u0,u1,P0,P1)=self.fit_QDA(X,y)
        correct=0.0;
        incorrect=0.0;
        (a,b)=np.shape(X)
        for i in range(np.shape(X)[0]):
            x0=X.iloc[i:i+1,0:b]
            ans=self.predict_QDA(CovI0,CovI1,det0,det1,u0,u1,P0,P1,x0)
            if ans==y.iloc[i]:
                correct+=1
            else:
                incorrect+=1
        return (correct/(incorrect+correct))
    def predict_k(self,X,y,k):
        num = int((np.shape(X)[0]))
        foldLength=int((num/k))
        myl=[]
        
        for a in range(0,k*foldLength,foldLength):
            training = X.drop(X.index[a:a + foldLength])
            resTraining = y.drop(y.index[a:a + foldLength]) #actual results
            testInp = X.iloc[a: a + foldLength] #fold for testing
            testOutp = y.iloc[a: a + foldLength] #result of fold for testing
            (CovI,u0,u1,P0,P1)=self.fit(training,resTraining)
            myl.append(self.predict_A(testInp,testOutp,CovI,u0,u1,P0,P1))
        return myl
    def predict_k_Log_odds(self,X,y,k):
        num = int((np.shape(X)[0]))
        foldLength=int((num/k))
        myl=[]
        
        for a in range(0,k*foldLength,foldLength):
            training = X.drop(X.index[a:a + foldLength])
            resTraining = y.drop(y.index[a:a + foldLength]) #actual results
            testInp = X.iloc[a: a + foldLength] #fold for testing
            testOutp = y.iloc[a: a + foldLength] #result of fold for testing
            (CovI,u0,u1,P0,P1)=self.fit(training,resTraining)
            myl.append(self.predict_A_Log_odds(testInp,testOutp,CovI,u0,u1,P0,P1))
        return myl
    def predict_k_QDA(self,X,y,k):
        num = int((np.shape(X)[0]))
        foldLength=int((num/k))
        myl=[]
        
        for a in range(0,k*foldLength,foldLength):
            training = X.drop(X.index[a:a + foldLength])
            resTraining = y.drop(y.index[a:a + foldLength]) #actual results
            testInp = X.iloc[a: a + foldLength] #fold for testing
            testOutp = y.iloc[a: a + foldLength] #result of fold for testing
            (CovI0,CovI1,det0,det1,u0,u1,P0,P1)=self.fit_QDA(training,resTraining)
            myl.append(self.predict_A_QDA(testInp,testOutp,CovI0,CovI1,det0,det1,u0,u1,P0,P1))
        return myl
    def predict(self,CovI,u0,u1,P0,P1,x0):
        #(CovI,u0,u1,P0,P1)=self.fit(X,y)
        a=x0.dot(CovI).dot(u0)-0.5*u0.T.dot(CovI).dot(u0)+P0
        a=a.values.astype(float)
        a=a[0]
        b=x0.dot(CovI).dot(u1)-0.5*u1.T.dot(CovI).dot(u1)+P1
        b=b.values.astype(float)
        b=b[0]
        if (a>=b):
            return 0
        else:
            return 1
    def predict_Log_odds(self,CovI,u0,u1,P0,P1,x0):
        #(CovI,u0,u1,P0,P1)=self.fit(X,y)
        a=x0.dot(CovI).dot(u1-u0)+-0.5*u1.T.dot(CovI).dot(u1)+0.5*u0.T.dot(CovI).dot(u0)+P1-P0
        a=a.values.astype(float)
        a=a[0]
        
        if (a>0):
            return 1
        else:
            return 0    
    def predict_QDA(self,CovI0,CovI1,det0,det1,u0,u1,P0,P1,x0):
        #(CovI,u0,u1,P0,P1)=self.fit(X,y)
        a=-0.5*(x0-u0.T).dot(CovI0).dot((x0-u0.T).T)+P0+det0
        a=a.values.astype(float)
        a=a[0]
        b=-0.5*(x0-u1.T).dot(CovI1).dot((x0-u1.T).T)+P1+det1
        b=b.values.astype(float)
        b=b[0]
        if (a>=b):
            return 0
        else:
            return 1
        
    def calculate_covariance_matrix(self, X):
            u=X.mean(axis=0)
            #u=u.mean(axis=0)
            Y = (X- u)
            n_samples = np.shape(X)[0]
            covariance_matrix = Y.T.dot(Y)
            return covariance_matrix
    

    def fit(self, sample, y):
        sample_0 = sample[y == 0]
        sample_1 = sample[y == 1]
        n_samples = np.shape(sample)[0]
        u0=sample_0.mean(axis=0)
        
        u1=sample_1.mean(axis=0)

        cVariance_0 = self.calculate_covariance_matrix(sample_0)
        cVariance_1 = self.calculate_covariance_matrix(sample_1)

        cVariance = cVariance_0 + cVariance_1
        cVariance = cVariance/(n_samples - 2)
    
        cVarianceInverse = pd.DataFrame(np.linalg.pinv(cVariance.values.astype(float)), cVariance.columns, cVariance.index)
        
        
        
        n_sample0=np.shape(sample_0)[0]
        n_sample1=np.shape(sample_1)[0]
        
        P1=math.log(n_sample1/n_samples)
        P0=math.log(n_sample0/n_samples)
        return (cVarianceInverse,u0,u1,P0,P1)
        
    def fit_QDA(self, sample, y):
        sample_0 = sample[y == 0]
        sample_1 = sample[y == 1]
        n_samples = np.shape(sample)[0]
        n_sample0=np.shape(sample_0)[0]
        n_sample1=np.shape(sample_1)[0]
        u0=sample_0.mean(axis=0)
        
        u1=sample_1.mean(axis=0)

        cVariance_0 = (self.calculate_covariance_matrix(sample_0))/(n_sample0-1)
        cVariance_1 = (self.calculate_covariance_matrix(sample_1))/(n_sample1-1)

        cVarianceInverse1 = pd.DataFrame(np.linalg.pinv(cVariance_1.values.astype(float)), cVariance_1.columns, cVariance_1.index)
        cVarianceInverse0 = pd.DataFrame(np.linalg.pinv(cVariance_0.values.astype(float)), cVariance_0.columns, cVariance_0.index)
        
        det1=-0.5*math.log(np.linalg.det(cVariance_1.values.astype(float)))
        det0=-0.5*math.log(np.linalg.det(cVariance_0.values.astype(float)))
        
        
        P1=math.log(n_sample1/n_samples)
        P0=math.log(n_sample0/n_samples)
        return (cVarianceInverse0,cVarianceInverse1,det0,det1,u0,u1,P0,P1)
        