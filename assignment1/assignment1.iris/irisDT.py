import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

encoder_x = LabelEncoder()

data = pd.read_csv("../../data set/Forest Fires Data Set/iris.csv")
# print(data)

X = data.iloc[:, 0:4]

y = data.iloc[:, 4]
y = np.array(y)
y = y.flatten()

scaler1 = preprocessing.StandardScaler()
scaler1.fit(X)
X = scaler1.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# print(train_X)
# print(train_Y)
# print(test_X)
# print(test_Y)

from sklearn import tree
from sklearn.metrics import classification_report, plot_confusion_matrix

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=40, random_state=0)
clf.fit(X_train, y_train)
y_pre = clf.predict(X_test)
# pre_train_Y = clf.predict(train_X)
tree.plot_tree(clf)
print(clf.score(X_test, y_test))
print(classification_report(y_test, y_pre, target_names=None))
# print(classification_report(train_Y, pre_train_Y, target_names=None))

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf, X_test, y_test,
                                 display_labels=encoder_x.classes_,
                                 cmap=plt.cm.Reds,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
