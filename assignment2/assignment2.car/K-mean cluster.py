import pandas as pd
import numpy as np
import math
from time import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Supervised Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn import tree, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# Clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

# Decomposition
from sklearn.decomposition import PCA, FastICA

#feature selection
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline

#Evaluation Metrics
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import classification_report, plot_confusion_matrix, silhouette_score
from sklearn import metrics


#load data
encoder_x =LabelEncoder()

data = pd.read_csv("../../data set/Car Evaluation/car.csv")
data.iloc[:,6] = encoder_x.fit_transform(data.iloc[:,6])
data = data.replace(999,np.NaN)
data = data.dropna()
# print(data.iloc[9:13, 0:3])

X = data.iloc[:,0:6]

y = data.iloc[:,6]
y = np.array(y)
y = y.flatten()
print(len(y),sum(y),encoder_x.classes_)#print the nnumber of instances, features, and the name of classes
scaler1 = preprocessing.MinMaxScaler()
scaler1.fit(X)
X = scaler1.transform(X)


#decide k -- Elbow method
distance = []
k = []
kmax = 25
#簇的数量
for n_clusters in range(1,kmax):
    cls = KMeans(n_clusters).fit(X)

    #曼哈顿距离
    def manhattan_distance(x,y):
        return np.sum(abs(x-y))

    distance_sum = 0
    for i in range(n_clusters):
        group = cls.labels_ == i
        members = X[group,:]
        for v in members:
            distance_sum += manhattan_distance(np.array(v), cls.cluster_centers_[i])
            print(i)
    distance.append(distance_sum)
    k.append(n_clusters)

plt.title("Elbow Method")
plt.scatter(k, distance)
plt.plot(k, distance)
plt.xlabel("k")
plt.ylabel("distance")
plt.show()



sil = []
k = []
set = []


# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for n_clusters in range(2, kmax+1):
  kmeans = KMeans(n_clusters = n_clusters).fit(X)
  labels = kmeans.labels_
  sil.append(silhouette_score(X, labels, metric = 'euclidean'))
  k.append(n_clusters)
  set.append([n_clusters,silhouette_score(X, labels, metric = 'euclidean')])


def getMax(list):
    maxX = 0;
    maxY = 0;
    for x in list:
        if (maxY < x[1]):
            maxY = x[1]
            maxX = x[0]
    return [maxX, maxY]

max_sil = getMax(set)
print ("test_max_alpha" + str(max_sil))

plt.title("Silhouette Method")
plt.scatter(k, sil)
plt.plot(k, sil)
plt.xlabel("k")
plt.ylabel(" silhouette value")
plt.text(max_sil[0],max_sil[1],(str(max_sil[0])+ "," + "{0:.4f}".format(max_sil[1])))
plt.show()