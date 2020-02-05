import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("../../data set/Car Evaluation/car.csv")
# data.iloc[:, 6] = encoder_x6.fit_transform(data.iloc[:, 6])
# encoder_x6.fit_transform(data.iloc[:, 6])


X = data.iloc[:, 0:6]

y = data.iloc[:, 6]
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

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix

# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
test = []

for k in np.arange(0, 0.5, 0.01):
    clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=40, random_state=0, ccp_alpha=k),
                             n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    y_pre = clf.predict(X_test)

    test.append(clf.score(X_test,y_test))
    print(clf.score(X_test, y_test))
    np.set_printoptions(precision=2)

plt.figure()
plt.plot(np.arange(0, 0.5, 0.01),test,"#8B27CC")
plt.text(np.argmax(test),np.max(test),("x = " +str(np.max(test))+ "y = " + str(np.argmax(test))))
plt.ylabel("Accuracy")
plt.xlabel("ccp_alpha")
plt.legend(['Test Data', 'Train Data'],  loc=0, borderaxespad=0.2)
plt.title("relation between Accuracy vs Pruning")
plt.show()

# clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=40, random_state=0,ccp_alpha = 0.0),n_estimators=100, random_state=0)
# clf.fit(X_train, y_train)
# y_pre = clf.predict(X_test)
# # pre_train_Y = clf.predict(train_X)
#
# print(clf.score(X_test, y_test))
# print(classification_report(y_test, y_pre,target_names = None))
# # print(classification_report(train_Y, pre_train_Y, target_names=None))
#
# np.set_printoptions(precision=2)

# # Plot non-normalized confusion matrix
# titles_options = [("Confusion matrix, without normalization", None),
#                   ("Normalized confusion matrix", 'true')]
# for title, normalize in titles_options:
#     disp = plot_confusion_matrix(clf, X_test, y_test,
#                                  display_labels=None,
#                                  cmap=plt.cm.Blues,
#                                  normalize=normalize)
#     disp.ax_.set_title(title)
#
#     print(title)
#     print(disp.confusion_matrix)
#
# plt.show()
