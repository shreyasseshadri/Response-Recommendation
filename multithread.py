'''
    Using Multithreading to optimize finding similiar response
    requires argument - cross_validation csv file
'''
import pandas as pd
import numpy as np
import queue
import sys
import multiprocessing as mp
from threading import Thread
from multiprocessing import Process
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data=pd.read_csv('processed.csv', lineterminator = '\n').dropna()
sentances_train=list(data['Contents'])
maxig=[]
maxpg=[]
q1=mp.Queue()
q2=mp.Queue()
def bubbleSort():
    global maxpg,maxig
    for passnum in range(len(maxig)-1,0,-1):
        for i in range(passnum):
            if maxpg[i]<maxpg[i+1]:
                temp = maxpg[i]
                temp1=maxig[i]
                maxpg[i] = maxpg[i+1]
                maxig[i] = maxig[i+1]
                maxpg[i+1] = temp
                maxig[i+1] = temp1

            
# Iterates through a subset of training data to find similiar elements
def printrecommandations(sentance_test,start,end):
    global maxig,maxpg,q1,q2
    maxi=[None]*3
    maxp=[0]*3
    for j in range(start,end):
        if(len(sentances_train[j])>1):
            tuple1=(sentance_test,sentances_train[j])
            tf_vectorizer = TfidfVectorizer(analyzer='word',stop_words='english')
            tf_matrix = tf_vectorizer.fit_transform(tuple1)
            result_cos = cosine_similarity(tf_matrix[0:1],tf_matrix)
            p=int(result_cos[0][1]*100)
            if p in maxp:
                continue
            if p>maxp[0]:
                maxp[2]=maxp[1]
                maxp[1]=maxp[0]
                maxp[0]=p
                maxi[2]=maxi[1]
                maxi[1]=maxi[0]
                maxi[0]=j
            elif p>maxp[1]:
                maxp[2]=maxp[1]
                maxp[1]=p
                maxi[2]=maxi[1]
                maxi[1]=j
            elif p>maxp[2]:
                maxp[2]=p
                maxi[2]=j
    q1.put(maxp)
    q2.put(maxi)

    
def recommend(sentance_test):
    global maxpg,maxig,q1,q2
    # Run all process parallely
    t1=Process(target=printrecommandations,args=(sentance_test,0,int(len(sentances_train)/4)))
    t2=Thread(target=printrecommandations,args=(sentance_test,int(len(sentances_train)/4),int(len(sentances_train)/2)))
    t3=Thread(target=printrecommandations,args=(sentance_test,int(len(sentances_train)/2),int(len(sentances_train)*3/4)))
    t4=Process(target=printrecommandations,args=(sentance_test,int(len(sentances_train)*3/4),int(len(sentances_train))))
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    while(not q1.empty()):
        maxpg=maxpg+q1.get()
    while(not q2.empty()):
        maxig=maxig+q2.get()
    maxig = [x for _,x in sorted(zip(maxpg,maxig))][::-1]
    maxpg = list(sorted(maxpg))[::-1]
    print("Setence Test: \n",sentance_test)
    confidence = []
    query = []
    resp = []
    for i in range(3):
            confidence.append(maxpg[i])
            query.append(data['Queries'][maxig[i]])
            resp.append(data['Responses'][maxig[i]])
    maxpg.clear()
    maxig.clear()
    q1=mp.Queue()
    q2=mp.Queue()
    return [confidence,query,resp]

cross_validation=sys.argv[1]
test = pd.read_csv(cross_validation)


Confidence = []
Query1 = []
Query2 = []
Response = []
for Query in test.Queries:
    confidence, query, resp = recommend(Query)
    for i in range(3):
        if confidence[i]>30:
            print(confidence[i])
            print("-"*50)
            print('Query Matched:', query[i])
            print("="*90)
            print('Response Matched:', resp[i])
    Confidence.append(confidence)
    Query1.append(Query)
    Query2.append(query)
    Response.append(resp)

# Write the similiar responses to a csv file
df1 = pd.DataFrame({'Query':Query1, 'Match':Query2, 'Confidence':Confidence, 'Response':Response})
df1 = df1[['Query', 'Match', 'Confidence', 'Response']]
df1.to_csv('match.csv')

