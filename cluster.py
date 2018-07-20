'''
    Clustering Queries using Doc2Vec vectors using sklearn Implementation and NLTK's implementaion.
    arguemts - processed csv file of query and response
             - Number of Clusters
'''
import pandas as pd
import sys


from gensim.models import doc2vec
from collections import namedtuple


# Load data
data=pd.read_csv(sys.argv[1])
doc1 = data.Queries.to_list()

# Transform data (you can add more data preprocessing steps) 

docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in enumerate(doc1):
    words = text.lower().split()
    tags = [i]
    docs.append(analyzedDocument(words, tags))

# Train model (set min_count = 1, if you want the model to work with the provided example data set)

from gensim.models.doc2vec import Doc2Vec
model = Doc2Vec(alpha=0.025, min_alpha=0.025,size = 100, window =4,min_count = 1)  # use fixed learning rate
model.build_vocab(docs)
for epoch in range(30):
    model.train(docs,epochs=model.iter, total_examples=model.corpus_count)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay

# Get the sentence vectors after the model is trained.
infvec=[]
for i in range(len(doc1)):
    tokens=doc1[i].split()
    infvec.append(model.infer_vector(tokens))

# Here begins kmeans from sklearn
from sklearn.cluster import KMeans
import collections
def cluster_sentences(infer_vector, nb_of_clusters=5):
        kmeans = KMeans(n_clusters=nb_of_clusters)
        kmeans.fit(infer_vector)
        clusters = collections.defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
                clusters[label].append(i)
        return dict(clusters)

nclusters=sys.argv[2]
clusters = cluster_sentences(infvec, nclusters)
for cluster in range(nclusters):
    print("cluster ",cluster,":")
    for i,sentence in enumerate(clusters[cluster]):    
        print("\tsentence ",sentence,": ",doc1[sentence],"\n","="*90)


#kmeans from nltk library
import numpy as np
infvec=np.array(infvec)
from nltk.cluster import KMeansClusterer, cosine_distance
clusterer = KMeansClusterer(200, cosine_distance,repeats=20,avoid_empty_clusters=True)
clusters = clusterer.cluster(infvec, True, trace=True)

ans=[[]for _ in range(50)]
for i in range(len(infvec)):
    ans[clusterer.classify_vectorspace(infvec[i])].append(i)

# Print the clusters
for i in range(50):
    print('cluster',i)
    for j in ans[i]:
        print(doc1[j],'\n',"="*90)

