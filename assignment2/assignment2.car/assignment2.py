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
X = scaler1.transform(X)# Normalize the feature to ensure every feature is in he same range

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# decide k -- Hierarchical Method
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X)
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()


#decide k -- Elbow method
distance = []
k = []
#簇的数量
for n_clusters in range(1,19):
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
plt.scatter(k, distance)
plt.plot(k, distance)
plt.xlabel("k")
plt.ylabel("distance")
plt.show()
#
# # Evaluation helper function
#
# # homogeneity_score：score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling
# # completeness_score：score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling
# # v_measure_score： score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling
# # ARI：Similarity score between -1.0 and 1.0. Random labelings have an ARI close to 0.0. 1.0 stands for perfect match.
# # AMI： The AMI returns a value of 1 when the two partitions are identical (ie perfectly matched).
# #       Random partitions (independent labellings) have an expected AMI around 0 on average hence can be negative.
# # silhouette_score： The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.
# #        Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.
# labels = y
# def bench_k_means(estimator, name, data):
#     t0 = time()
#     estimator.fit(X)
#     print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
#           % (name, (time() - t0), estimator.inertia_,
#              metrics.homogeneity_score(labels, estimator.labels_),
#              metrics.completeness_score(labels, estimator.labels_),
#              metrics.v_measure_score(labels, estimator.labels_),
#              metrics.adjusted_rand_score(labels, estimator.labels_),
#              metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
#              metrics.silhouette_score(data, estimator.labels_,
#                                       metric='euclidean',
#                                       sample_size=sample_size)))
#
# data = X
# # Data decomposition
# # PCA
# # pca = PCA(n_components=100)
# # reduced_data_PCA = pca.fit_transform(data)
# #or select feature based on explained_variance
# pca = PCA(n_components=3)
# reduced_data_PCA = pca.fit_transform(data)
# print (pca.explained_variance_ratio_)
# print (pca.explained_variance_)
# print (pca.n_components_)
#
# #FastICA
# reduced_data_ICA = FastICA(n_components=2, random_state=0).fit_transform(data)
#
#
# #Other feature selection algorithm
# selector = SelectKBest(f_classif, k=2)
# X_new = selector.fit_transform(X, y)
# new_index = selector.get_support(indices = True)
# print(new_index)
#
# # Clustering, Evaluation on original dataset
# np.random.seed(42)
# data = X
# n_samples, n_features = data.shape
# # n_digits = len(np.unique(y))
# n_digits = 6
# sample_size = n_samples
#
# print("Real classes: %d, \t Number of samples %d, \t Number of features %d"
#       % (n_digits, n_samples, n_features))
#
# print('Evaluation for Kmeans on original dataset')
# print(82 * '_')
# print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
#
# bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
#               name="k-means++", data=data)
#
# bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
#               name="random", data=data)
#
# # in this case the seeding of the centers is deterministic, hence we run the
# # kmeans algorithm only once with n_init=1
# pca = PCA(n_components=n_digits).fit(data)
# bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
#               name="PCA-based",
#               data=data)
# print(82 * '_')
#
# # Clustering, Evaluation on PCA reduced dataset
# print('\nEvaluation for Kmeans on PCA reduced dataset')
# print(82 * '_')
# print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
# bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
#               name="k-means++", data=reduced_data_PCA)
#
# bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
#               name="random", data=reduced_data_PCA)
#
# # in this case the seeding of the centers is deterministic, hence we run the
# # kmeans algorithm only once with n_init=1
# pca = PCA(n_components=n_digits).fit(data)
# bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
#               name="PCA-based",
#               data=reduced_data_PCA)
# print(82 * '_')
#
# # Clustering, Evaluation on ICA reduced dataset
# print('\nEvaluation for Kmeans on ICA reduced dataset')
# print(82 * '_')
# print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
# bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
#               name="k-means++", data=reduced_data_ICA)
#
# bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
#               name="random", data=reduced_data_ICA)
#
# # # in this case the seeding of the centers is deterministic, hence we run the
# # # kmeans algorithm only once with n_init=1
# # pca = PCA(n_components=n_digits).fit(data)
# # bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
# #               name="PCA-based",
# #               data=reduced_data)
# print(82 * '_')
#
# # Clustering, Evaluation on anova selected dataset
# print('\nEvaluation for Kmeans on Anova selected dataset')
# print(82 * '_')
# print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
# bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
#               name="k-means++", data=X_new)
#
# bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
#               name="random", data=X_new)
#
# # # in this case the seeding of the centers is deterministic, hence we run the
# # # kmeans algorithm only once with n_init=1
# # pca = PCA(n_components=n_digits).fit(data)
# # bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
# #               name="PCA-based",
# #               data=reduced_data)
# print(82 * '_')
#
#
#
# # colors = ['darkorange','royalblue','c','m','y','#e24fff','#524C90','#845868']
# colors = plt.cm.Set1(np.linspace(0, 1, 10))
# n_clusters=n_digits
# # #############################################################################
# # Visualize the results on original data
# clf = KMeans(n_clusters=n_digits, random_state=0)
# y_pred = clf.fit_predict(data)
#
# cents = clf.cluster_centers_#质心
# labels = clf.labels_#样本点被分配到的簇的索引
#
# #画出聚类结果，每一类用一种颜色
# for i in range(n_clusters):
#     index = np.nonzero(labels==i)[0]
#     x0 = data[index,0]
#     x1 = data[index,1]
#     y_i = y[index]
#     for j in range(len(x0)):
#         plt.text(x0[j],x1[j],str(int(y_i[j])),color=colors[i],\
#                 fontdict={'weight': 'bold', 'size': 6})
# plt.title('K-means clustering on the original dataset\n')
#
# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = data[:, 0].min() - 0.1, data[:, 0].max() + 0.1
# y_min, y_max = data[:, 1].min() - 0.1, data[:, 1].max() + 0.1
#
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()
#
# # #############################################################################
# # Visualize the results on PCA reduced data
# clf_PCA = KMeans(n_clusters=n_digits, random_state=0)
# y_pred = clf_PCA.fit_predict(reduced_data_PCA)
#
# cents = clf_PCA.cluster_centers_#质心
# labels_PCA = clf_PCA.labels_#样本点被分配到的簇的索引
#
# #画出聚类结果，每一类用一种颜色
# for i in range(n_clusters):
#     index = np.nonzero(labels_PCA==i)[0]
#     x0 = reduced_data_PCA[index,0]
#     x1 = reduced_data_PCA[index,1]
#     y_i = y[index]
#     for j in range(len(x0)):
#         plt.text(x0[j],x1[j],str(int(y_i[j])),color=colors[i],\
#                 fontdict={'weight': 'bold', 'size': 6})
# plt.title('K-means clustering on the PCA reduced dataset\n')
#
# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = reduced_data_PCA[:, 0].min() - 0.1, reduced_data_PCA[:, 0].max() +0.1
# y_min, y_max = reduced_data_PCA[:, 1].min() - 0.1, reduced_data_PCA[:, 1].max()+0.1
#
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()
#
# # #############################################################################
# # Visualize the results on ICA reduced data
# clf_ICA = KMeans(n_clusters=n_digits, random_state=0)
# y_pred = clf_ICA.fit_predict(reduced_data_ICA)
#
# cents = clf_ICA.cluster_centers_#质心
# labels_ICA = clf_ICA.labels_#样本点被分配到的簇的索引
#
# for i in range(n_clusters):
#     index = np.nonzero(labels_ICA==i)[0]
#     x0 = reduced_data_ICA[index,0]
#     x1 = reduced_data_ICA[index,1]
#     y_i = y[index]
#     for j in range(len(x0)):
#         plt.text(x0[j],x1[j],str(int(y_i[j])),color=colors[i],\
#                 fontdict={'weight': 'bold', 'size': 6})
# plt.title('K-means clustering on the ICA reduced dataset\n')
#
# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = reduced_data_ICA[:, 0].min() - 0.01, reduced_data_ICA[:, 0].max() + 0.01
# y_min, y_max = reduced_data_ICA[:, 1].min() - 0.01, reduced_data_ICA[:, 1].max() + 0.01
#
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()
#
# #################################################################3
# # Visualize the results on Anova reduced data
# clf_new = KMeans(n_clusters=n_digits, random_state=0)
# y_pred = clf_new .fit_predict(X_new)
#
# cents = clf_new .cluster_centers_#质心
# labels_new = clf_new .labels_#样本点被分配到的簇的索引
#
# for i in range(n_clusters):
#     index = np.nonzero(labels_new==i)[0]
#     x0 = X_new[index,0]
#     x1 = X_new[index,1]
#     y_i = y[index]
#     for j in range(len(x0)):
#         plt.text(x0[j],x1[j],str(int(y_i[j])),color=colors[i],\
#                 fontdict={'weight': 'bold', 'size': 6})
# plt.title('K-means clustering on the Anova selected dataset\n')
#
# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = X_new[:, 0].min() - 0.1, X_new[:, 0].max() + 0.1
# y_min, y_max = X_new[:, 1].min() - 0.1, X_new[:, 1].max() + 0.1
#
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()

