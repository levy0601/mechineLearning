# import needed module
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
from sklearn.metrics import classification_report, plot_confusion_matrix
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

data = X
# Data decomposition
# PCA
# pca = PCA(n_components=100)
# reduced_data_PCA = pca.fit_transform(data)
#or select feature based on explained_variance
pca = PCA(n_components = 2)
reduced_data_PCA = pca.fit_transform(data)
print (pca.explained_variance_ratio_)
print (pca.explained_variance_)
print (pca.n_components_)

#FastICA
reduced_data_ICA = FastICA(n_components=4, random_state=0).fit_transform(data)

# Clustering, Evaluation on original dataset
np.random.seed(42)
data = X
n_samples, n_features = data.shape
# n_digits = len(np.unique(y))
n_digits = 4
sample_size = n_samples


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


# #############################################################################
# Visualize the results on PCA reduced data
clf_PCA = KMeans(n_clusters=n_digits, random_state=0)
y_pred = clf_PCA.fit_predict(reduced_data_PCA)

cents = clf_PCA.cluster_centers_#质心
labels_PCA = clf_PCA.labels_#样本点被分配到的簇的索引

#画出聚类结果，每一类用一种颜色
for i in range(n_clusters):
    index = np.nonzero(labels_PCA==i)[0]
    x0 = reduced_data_PCA[index,0]
    x1 = reduced_data_PCA[index,1]
    y_i = y[index]
    for j in range(len(x0)):
        plt.text(x0[j],x1[j],str(int(y_i[j])),color=colors[i],\
                fontdict={'weight': 'bold', 'size': 7})
plt.title('K-means clustering on the PCA reduced dataset (ad)\n')

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data_PCA[:, 0].min() - 0.1, reduced_data_PCA[:, 0].max() +0.1
y_min, y_max = reduced_data_PCA[:, 1].min() - 0.1, reduced_data_PCA[:, 1].max()+0.1

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

# #############################################################################
# Visualize the results on ICA reduced data
clf_ICA = KMeans(n_clusters=n_digits, random_state=0)
y_pred = clf_ICA.fit_predict(reduced_data_ICA)

cents = clf_ICA.cluster_centers_#质心
labels_ICA = clf_ICA.labels_#样本点被分配到的簇的索引

for i in range(n_clusters):
    index = np.nonzero(labels_ICA==i)[0]
    x0 = reduced_data_ICA[index,0]
    x1 = reduced_data_ICA[index,1]
    y_i = y[index]
    for j in range(len(x0)):
        plt.text(x0[j],x1[j],str(int(y_i[j])),color=colors[i],\
                fontdict={'weight': 'bold', 'size': 7})
plt.title('K-means clustering on the ICA reduced dataset (ad)\n')

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data_ICA[:, 0].min() - 0.01, reduced_data_ICA[:, 0].max() + 0.01
y_min, y_max = reduced_data_ICA[:, 1].min() - 0.01, reduced_data_ICA[:, 1].max() + 0.01

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()