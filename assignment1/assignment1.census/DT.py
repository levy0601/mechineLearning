import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


encoder_x =LabelEncoder()

data = pd.read_csv("../../data set/Census Income Data Set/adult.csv")

data.iloc[:,0] = encoder_x.fit_transform(data.iloc[:,0])
data.iloc[:,1] = encoder_x.fit_transform(data.iloc[:,1])
data.iloc[:,2] = encoder_x.fit_transform(data.iloc[:,2])
data.iloc[:,3] = encoder_x.fit_transform(data.iloc[:,3])
data.iloc[:,4] = encoder_x.fit_transform(data.iloc[:,4])
data.iloc[:,5] = encoder_x.fit_transform(data.iloc[:,5])
data.iloc[:,6] = encoder_x.fit_transform(data.iloc[:,6])
data.iloc[:,7] = encoder_x.fit_transform(data.iloc[:,7])
data.iloc[:,8] = encoder_x.fit_transform(data.iloc[:,8])
data.iloc[:,9] = encoder_x.fit_transform(data.iloc[:,9])
data.iloc[:,10] = encoder_x.fit_transform(data.iloc[:,10])
data.iloc[:,11] = encoder_x.fit_transform(data.iloc[:,11])
data.iloc[:,12] = encoder_x.fit_transform(data.iloc[:,12])
data.iloc[:,13] = encoder_x.fit_transform(data.iloc[:,13])



# print(data)

X = data.iloc[:, 0:14]

y = data.iloc[:, 14]
y = np.array(y)
y = y.flatten()

scaler1 = preprocessing.StandardScaler()
scaler1.fit(X)
X = scaler1.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=0)

# print(train_X)
# print(train_Y)
# print(test_X)
# print(test_Y)

from sklearn import tree
from sklearn.metrics import classification_report, plot_confusion_matrix

ccp_alpha = 0.05
print ("ccp_alpha = "  + str(ccp_alpha))
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=40, random_state=0,ccp_alpha = ccp_alpha)

clf.fit(X_train, y_train)
y_tra = clf.predict(X_train)
y_pre = clf.predict(X_test)
# pre_train_Y = clf.predict(train_X)
tree.plot_tree(clf)
print ("train score")
print(clf.score(X_train, y_train))
print ("test score")
print(clf.score(X_test, y_test))
print ("train report")
print(classification_report(y_train, y_tra, target_names=None))
print ("test report")
print(classification_report(y_test, y_pre, target_names=None))
# print(classification_report(train_Y, pre_train_Y, target_names=None))

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf, X_test, y_test,
                                 display_labels=None,
                                 cmap=plt.cm.Reds,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf, X_train, y_train,
                                 display_labels=None,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

clf = tree.DecisionTreeClassifier(random_state=0)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")

clfs = []
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
      clfs[-1].tree_.node_count, ccp_alphas[-1]))

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()

train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()

print("done")