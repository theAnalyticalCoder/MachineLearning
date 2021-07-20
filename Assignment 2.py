# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 12:29:19 2019

@author: liam
"""

import numpy as np
from numpy import genfromtxt
import pandas as pd
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import csv
import re    
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import tree

class DataPreprocess: 

    def __init__(self, TrainData, TestData,sub):
        self.TrainDataset = TrainData
        self.TestDataset = TestData
        self.comment = TrainData.iloc[:,1]
        
        self.TestComment = TestData.iloc[:,-1]
       
        self.subreddit=sub

       

    def ModelEvaluation(self, Dataset, Output, TestSet, TestOutput, Model):


        if (Model == "LR"):
            model = LogisticRegression().fit(Dataset, Output)
        elif (Model == "NB"):
            model = MultinomialNB().fit(Dataset, Output)
        elif (Model == "SVC"):
            model = LinearSVC(random_state=0, tol=1e-5, fit_intercept=True,
                                loss='squared_hinge').fit(Dataset, Output)
        elif (Model == "DTC"):
            model = tree.DecisionTreeClassifier(random_state=0).fit(Dataset, Output)

        predictions = model.predict(TestSet)
        
        counter = 0
        #prediction output of test.csv file
        '''
        with open('output.csv','w') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(['Id','Category'])
                for x in predictions:
                        row = [str(counter), x]
                        writer.writerow(row)
                        counter += 1
        csvFile.close()
        '''
        #print( model.predict(TestSet)) #predictions of LR in an array
        print(Model, ":", (model.score(TestSet, TestOutput) * 100)) #accuracy of predictions
        
    def get_prediction(self, Dataset, Output, TestSet, Model):


        if (Model == "LR"):
            model = LogisticRegression().fit(Dataset, Output)
        elif (Model == "NB"):
            model = MultinomialNB().fit(Dataset, Output)
        elif (Model == "SVC"):
            model = LinearSVC(random_state=0, tol=1e-5, fit_intercept=True,
                                loss='squared_hinge').fit(Dataset, Output)
        elif (Model == "DTC"):
            model = tree.DecisionTreeClassifier(random_state=0).fit(Dataset, Output)

        predictions = model.predict(TestSet)
        
        counter = 0
        return predictions
        #print( model.predict(TestSet)) #predictions of LR in an array
        #print(Model, ":", (model.score(TestSet, TestOutput) * 100)) #accuracy of predictions


       


class LemmaTokenizer(object): #lemmatizer from nltk
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

        
class Stemmer(object): #porterstemmer from nltk
    def __init__(self):
        self.wnl = PorterStemmer()
    def __call__(self, articles):
        return [self.wnl.stem(t) for t in word_tokenize(articles)]



reddit_test = pd.read_csv('reddit_test.csv')#, sep=',',header=None)
reddit_train = pd.read_csv('reddit_train.csv')#, sep=',',header=None)

'''word_list = list() #will hold all words from the document, will be used to generate stopwords 

comment=reddit_train.iloc[1:,1]

counter = 0
for i in comment:
    word_row=i.split(" ")
    for j in word_row:
        word_list.append(j)
        counter+=1

d = {}
e={}
for x in word_list:
    
    match=re.match('[0-9]',x)
    
    if match:
        if x.lower() not in d.keys():
            d[x.lower()] = 1
        else: 
            d[x.lower()] += 1
    x=re.sub('[^a-zA-Z]+', '', x)#line that gets rid of non alphabetical stuff
    if (len(x) <= 4): 
        if x.lower() not in e.keys():
            e[x.lower()] = 1
        else: 
            e[x.lower()] += 1


k = Counter(d)
k2 = Counter(e)
high = k2.most_common(50) #3 most common words removed
low = k.most_common()[:-6625-1:-1]

words_to_remove = []
least_common = []
for x in high:
    words_to_remove.append(x[0])

for x in low:
    words_to_remove.append(x[0])

#print(words_to_remove)
#print(least_common)


'''
a=reddit_train.iloc[:,-1].replace({'nba': 'sports','nhl':'sports','nfl':'sports', 'hockey':'sports','baseball':'sports',
                        'soccer': 'sports','Overwatch':'games','GlobalOffensive':'games','leagueoflegends':'games','wow':'games',
                         'canada':'pol', 'worldnews':'pol','europe':'pol', 'conspiracy':'pol', 'funny':'misc', 'AskReddit':'misc',
                         'Music':'misc', 'trees':'misc','movies':'misc','anime':'misc','gameofthrones':'misc'}, regex=True)
four_cat=reddit_train.iloc[:,0:2]
four_cat['subreddit']=a
obj = DataPreprocess(reddit_train, reddit_test,a)
obj_all_subs = DataPreprocess(reddit_train, reddit_test,reddit_train.iloc[:,-1])
tfidf = TfidfVectorizer( min_df=2, max_df=2600)
#|'nfl'|'soccer'|'baseball')
#TrainX, TestX, TrainY, TestY = train_test_split(obj_all_subs.comment, obj_all_subs.subreddit, test_size=0.20, random_state=4)
#TrainX_S, TestX_S, TrainY_S, TestY_S = train_test_split(obj.comment, obj.subreddit, test_size=0.20, random_state=4)

print("------------------------Sports-----------------------------")
g=reddit_train[(reddit_train['subreddits']==('hockey'))]
g=g.append(reddit_train[(reddit_train['subreddits']==('nfl'))])
g=g.append(reddit_train[(reddit_train['subreddits']==('soccer'))])
g=g.append(reddit_train[(reddit_train['subreddits']==('baseball'))])
sports=g.append(reddit_train[(reddit_train['subreddits']==('nba'))])
TrainX_S, TestX_S, TrainY_S, TestY_S = train_test_split(sports.iloc[:,1], sports.iloc[:,-1], test_size=0.20, random_state=4)
#TrainX_S=sports.iloc[0:int(0.8*len(sports)),1]
#TrainY_S=sports.iloc[0:int(0.8*len(sports)),-1]
#TestX_S=sports.iloc[int(0.8*len(sports)):,1]
#TestY_S=sports.iloc[int(0.8*len(sports)):,-1]
TRX = tfidf.fit_transform(TrainX_S)
TX = tfidf.transform(TestX_S)
RX = tfidf.transform(obj.TestComment)
obj.ModelEvaluation(TRX,TrainY_S,TX,TestY_S, "NB") #regular testing LR
Pred_S=obj.get_prediction(TRX,TrainY_S,RX, "NB")
print("------------------------Games-----------------------------")

g=reddit_train[(reddit_train['subreddits']==('Overwatch'))]
g=g.append(reddit_train[(reddit_train['subreddits']==('GlobalOffensive'))])
g=g.append(reddit_train[(reddit_train['subreddits']==('leagueoflegends'))])
games=g.append(reddit_train[(reddit_train['subreddits']==('wow'))])
TrainX_G, TestX_G, TrainY_G, TestY_G = train_test_split(games.iloc[:,1], games.iloc[:,-1], test_size=0.20, random_state=4)
#TrainX_G=games.iloc[:,1]
#TrainY_G=games.iloc[:,-1]
TRX = tfidf.fit_transform(TrainX_G)
TX = tfidf.transform(TestX_G)
RX = tfidf.transform(obj.TestComment)
obj.ModelEvaluation(TRX,TrainY_G,TX,TestY_G, "NB")
Pred_G=obj.get_prediction(TRX,TrainY_G,RX, "NB")
print("------------------------pol-----------------------------")

g=reddit_train[(reddit_train['subreddits']==('canada'))]
g=g.append(reddit_train[(reddit_train['subreddits']==('europe'))])
g=g.append(reddit_train[(reddit_train['subreddits']==('conspiracy'))])
pol=g.append(reddit_train[(reddit_train['subreddits']==('worldnews'))])
TrainX_P, TestX_P, TrainY_P, TestY_P = train_test_split(pol.iloc[:,1], pol.iloc[:,-1], test_size=0.20, random_state=4)
#TrainX_P=pol.iloc[:,1]
#TrainY_P=pol.iloc[:,-1]
TRX = tfidf.fit_transform(TrainX_P)
TX = tfidf.transform(TestX_P)
RX = tfidf.transform(obj.TestComment)
obj.ModelEvaluation(TRX,TrainY_P,TX,TestY_P, "NB")
Pred_P=obj.get_prediction(TRX,TrainY_P,RX, "NB")

print("------------------------MISC-----------------------------")
g=reddit_train[(reddit_train['subreddits']==('funny'))]
g=g.append(reddit_train[(reddit_train['subreddits']==('AskReddit'))])
g=g.append(reddit_train[(reddit_train['subreddits']==('trees'))])
g=g.append(reddit_train[(reddit_train['subreddits']==('Music'))])
g=g.append(reddit_train[(reddit_train['subreddits']==('anime'))])
g=g.append(reddit_train[(reddit_train['subreddits']==('gameofthrones'))])
misc=g.append(reddit_train[(reddit_train['subreddits']==('movies'))])
TrainX_M, TestX_M, TrainY_M, TestY_M = train_test_split(misc.iloc[:,1], misc.iloc[:,-1], test_size=0.20, random_state=4)
#TrainX_M=misc.iloc[:,1]
#TrainY_M=misc.iloc[:,-1]
TRX = tfidf.fit_transform(TrainX_M)
TX = tfidf.transform(TestX_M)
RX = tfidf.transform(obj.TestComment)
obj.ModelEvaluation(TRX,TrainY_M,TX,TestY_M, "NB")
Pred_M=obj.get_prediction(TRX,TrainY_M,RX, "NB")
'''
print("------------------------intr-----------------------------")
g=reddit_train[(reddit_train['subreddits']==('Music'))]
g=g.append(reddit_train[(reddit_train['subreddits']==('anime'))])
g=g.append(reddit_train[(reddit_train['subreddits']==('gameofthrones'))])
intr=g.append(reddit_train[(reddit_train['subreddits']==('movies'))])
TrainX_I, TestX_I, TrainY_I, TestY_I = train_test_split(intr.iloc[:,1], intr.iloc[:,-1], test_size=0.20, random_state=4)
#TrainX_M=misc.iloc[:,1]
#TrainY_M=misc.iloc[:,-1]
TRX = tfidf.fit_transform(TrainX_I)
TX = tfidf.transform(TestX_I)
RX = tfidf.transform(obj.TestComment)
obj.ModelEvaluation(TRX,TrainY_I,TX,TestY_I, "NB")
Pred_M=obj.get_prediction(TRX,TrainY_I,RX, "NB")
'''
print("------------------------Main 4 split-----------------------------")
TrainX, TestX, TrainY, TestY = train_test_split(reddit_train.iloc[:,1], a, test_size=0.20, random_state=4)
#TrainX=obj.comment
#TrainY=obj.subreddit
RealTestX = obj.TestComment
TRX = tfidf.fit_transform(TrainX)
TX = tfidf.transform(TestX)
RX = tfidf.transform(obj.TestComment)
obj.ModelEvaluation(TRX,TrainY,TX,TestY, "NB")
Pred_4=obj.get_prediction(TRX,TrainY,RX, "NB")
answers=list()
counter=0
for (sub,sports,games,pol,misc) in zip(Pred_4,Pred_S,Pred_G,Pred_P,Pred_M):
    sub=str(sub)
    if sub=='sports':
        answers.append(str(sports))
    elif sub=='games':
        answers.append(str(games))
    elif sub=='pol':
        answers.append(str(pol))
    elif sub=='misc':
        answers.append(str(misc))
counter = 0
        #prediction output of test.csv file
        
with open('test1.csv','w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['Id','Category'])
    for x in answers:
        row = [str(counter), x]
        writer.writerow(row)
        counter += 1
csvFile.close()
#specify TF for tfidf

#s=pd.Series()

#w=TestY.where(TestY=='sports')
#maybe dont include the lemmatization since it seems to do more bad
'''tokenizer=LemmaTokenizer(),'''
'''for i in range(100,5000,100):
    tfidf = TfidfVectorizer( min_df=2, max_df=i) #max_df=1210
#stop_words=words_to_remove, , lowercase=True,

    vectorizer = CountVectorizer(stop_words=words_to_remove)

    TfOrCV = "TF"
    print("-----------------------------------------------------")
    if (TfOrCV == "TF"): #specify TF for tfidf
        TRX = tfidf.fit_transform(TrainX)
        TX = tfidf.transform(TestX)
        RX = tfidf.transform(obj.TestComment)
    elif (TfOrCV == "CV"): #specifc CV for Count Vectorization
        TrainX = vectorizer.fit_transform(TrainX)
        TestX = vectorizer.transform(TestX)
        RealTest = vectorizer.transform(obj.TestComment)
        
#custom stops words, min_df = 2, max_df = 0.0185, lowercase=TrueTrained on entire dataset 

#obj.ModelEvaluation(TrainX,TrainY,RealTest,TestY, "LR") #Real test set LR
    print(i)
    obj.ModelEvaluation(TRX,TrainY,TX,TestY, "LR") #regular testing LR
#obj.ModelEvaluation(TRX,TrainY,RX,TestY, "NB") #Real test set NB scikit
    obj.ModelEvaluation(TRX,TrainY,TX,TestY, "NB") #regular testing NB scikit
'''
 #regular testing NB scikit  

      
#custom stops words, min_df = 2, max_df = 0.0185, lowercase=TrueTrained on entire dataset 
a=3
#obj.ModelEvaluation(TrainX,TrainY,RealTest,TestY, "LR") #Real test set LR
    
#obj.ModelEvaluation(TRX,TrainY,TX,TestY, "LR") #regular testing LR
#obj.ModelEvaluation(TRX,TrainY,RX,TestY, "NB") #Real test set NB scikit
#obj.ModelEvaluation(TRX,TrainY,TX,TestY, "NB") #regular testing NB scikit
#obj.ModelEvaluation(TrainX,TrainY,RealTest,TestY, "SVC") #Real test set NB scikit
#obj.ModelEvaluation(TrainX,TrainY,TestX,TestY, "SVC") #regular testing NB scikit
#obj.ModelEvaluation(TrainX,TrainY,RealTest,TestY, "DTC") #Real test set NB scikit
#obj.ModelEvaluation(TrainX,TrainY,TestX,TestY, "DTC") #regular testing NB scikit


#Best Trial so far min_df=2, max_df=0.025 test_size=0.05 55.233% on kaggle

#Best Trial on held out test set 55.32% on a 15% held out test set, no tokenizer, min_df=2, max_df=0.025

#best trial on held out set min_df = 1, max_df = 1210, 56%