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
kmax = 10
#簇的数量
# for n_clusters in range(1,kmax):
#     cls = KMeans(n_clusters).fit(X)
#
#     #曼哈顿距离
#     def manhattan_distance(x,y):
#         return np.sum(abs(x-y))
#
#     distance_sum = 0
#     for i in range(n_clusters):
#         group = cls.labels_ == i
#         members = X[group,:]
#         for v in members:
#             distance_sum += manhattan_distance(np.array(v), cls.cluster_centers_[i])
#             print(i)
#     distance.append(distance_sum)
#     k.append(n_clusters)

for n_clusters in range(1,kmax):
    cls = KMeans(n_clusters).fit(X)
    distance.append(cls.inertia_)#Sum of squared distances of samples to their closest cluster center.
    k.append(n_clusters)

plt.title("Car : Elbow Method")
plt.scatter(k, distance)
plt.plot(k, distance)
plt.xlabel("k")
plt.ylabel(" squared distances of samples to their closest cluster center")
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

plt.title("Car: Silhouette Method")
plt.scatter(k, sil)
plt.plot(k, sil)
plt.xlabel("k")
plt.ylabel(" silhouette value")
plt.text(max_sil[0],max_sil[1],(str(max_sil[0])+ "," + "{0:.4f}".format(max_sil[1])))
plt.show()



colors = plt.cm.Set1(np.linspace(0, 1, 10))
# https://matplotlib.org/examples/color/colormaps_reference.html
data = X
n_digits = 2
n_clusters = n_digits
# #############################################################################
# Visualize the results on original data
clf = KMeans(n_clusters=n_digits, random_state=0)
y_pred = clf.fit_predict(data)

cents = clf.cluster_centers_#质心
labels = clf.labels_#样本点被分配到的簇的索引

#画出聚类结果，每一类用一种颜色
for i in range(n_clusters):
    index = np.nonzero(labels==i)[0]
    x0 = data[index,0]
    x1 = data[index,1]
    y_i = y[index]
    for j in range(len(x0)):
        plt.text(x0[j],x1[j],str(int(y_i[j])),color=colors[i],\
                fontdict={'weight': 'bold', 'size': 7})
plt.title('K-means clustering on the original dataset (ad)\n')

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = data[:, 0].min() - 0.1, data[:, 0].max() + 0.1
y_min, y_max = data[:, 1].min() - 0.1, data[:, 1].max() + 0.1

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()