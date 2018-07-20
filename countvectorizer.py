'''
  This is an earlier approach commented for reference purposes
'''

# import pandas as pd

# data=pd.read_csv('data.csv')
# sentances_train=list(data['Queries'])

# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# def printrecommandations(sentance_test):
#     maxpercent=0
#     maxindex=-1
#     for j in range(len(sentances_train)):
#         if(len(sentances_train[j])>1):
#             tuple1=(sentance_test,sentances_train[j])
#             count_vectorizer = CountVectorizer(analyzer='word',stop_words='english')
#             count_matrix = count_vectorizer.fit_transform(tuple1)
#             result_cos = cosine_similarity(count_matrix[0:1],count_matrix)
#             if(result_cos[0][1]*100>maxpercent):
#                 maxpercent=result_cos[0][1]*100
#                 maxindex=j
#     print("Setence Test: \n",sentance_test)
#     print(maxpercent)
#     print("-"*50)
#     print("Sentence Matched \n",data['Queries'][maxindex])
#     print("Sentence Matched \n",data['Responses'][maxindex])
#     print("="*90)


# count_vectorizer = CountVectorizer(analyzer='word',stop_words='english')
# count_matrix = count_vectorizer.fit_transform(sen)


#--------------------------------------------------------------------------------------

'''
    This is the approach that uses cosine similarity between Tf-idf vectors to get the most similiar response
'''
import pickle

with open('tfidf.pkl', 'rb') as pickle_file:
    tfidf = pickle.load(pickle_file)
with open('count_vec.pkl', 'rb') as pickle_file:
    count_vect = pickle.load(pickle_file)
import pandas as pd
data=pd.read_csv("processed.csv")

from sklearn.cross_validation import train_test_split
X_train,X_test=train_test_split(data,test_size=0.3)

from sklearn.metrics.pairwise import cosine_similarity
def get_vector(sentence):
    doc_freq_term = count_vect.transform([sentence])
    return tfidf.transform(doc_freq_term)
def cosine(str1,str2):
    return cosine_similarity(get_vector(str1),get_vector(str2))

def printrecommandations(sentance_test):
    maxpercent=0
    maxindex=-1
    for j,v in X_train['Contents'].items():
        cos=cosine(sentance_test,v)[0][0]
        if(cos>maxpercent):
            maxpercent=cos
            maxindex=j
    if True:    
        print("Setence Test: \n",sentance_test)
        print(maxpercent*100)
        print("-"*50)
        print("Sentence Matched \n",data['Contents'][maxindex])
        print("Sentence Matched \n",data['Responses'][maxindex])
        print("="*90)        
