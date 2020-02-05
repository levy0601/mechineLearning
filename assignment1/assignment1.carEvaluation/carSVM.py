import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


data = pd.read_csv("../../data set/Car Evaluation/car.csv")

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

from sklearn import svm, metrics
from sklearn.metrics import classification_report, plot_confusion_matrix
result = []
kernel = "linear"
clf = svm.SVC(kernel=kernel,decision_function_shape='ovo')
clf.fit(X_train, y_train)
y_pre_ovo = clf.predict(X_test)
result.append(clf.score(X_test, y_test))
# pre_train_Y = clf.predict(train_X)


print(kernel + " " + str( clf.score(X_test, y_test)))

kernel = "poly"
clf = svm.SVC(kernel=kernel,decision_function_shape='ovo')
clf.fit(X_train, y_train)
y_pre_ovo = clf.predict(X_test)
result.append(clf.score(X_test, y_test))
# pre_train_Y = clf.predict(train_X)

print(kernel + " " + str( clf.score(X_test, y_test)))


kernel = "rbf"
clf = svm.SVC(kernel=kernel,decision_function_shape='ovo')
clf.fit(X_train, y_train)
y_pre_ovo = clf.predict(X_test)
result.append(clf.score(X_test, y_test))
# pre_train_Y = clf.predict(train_X)

print(kernel + " " + str( clf.score(X_test, y_test)))


kernel = "sigmoid"
clf = svm.SVC(kernel=kernel,decision_function_shape='ovo')
clf.fit(X_train, y_train)
y_pre_ovo = clf.predict(X_test)
result.append(clf.score(X_test, y_test))
# pre_train_Y = clf.predict(train_X)

print(kernel + " " + str( clf.score(X_test, y_test)))



# print(classification_report(y_test, y_pre_ovo, target_names = None))
# print(classification_report(train_Y, pre_train_Y, target_names=None))

np.set_printoptions(precision=2)

# # Plot non-normalized confusion matrix
# titles_options = [("Confusion matrix, without normalization", None),
#                   ("Normalized confusion matrix", 'true')]
# for title, normalize in titles_options:
#     disp = plot_confusion_matrix(clf, X_test, y_test,
#                                  display_labels= None,
#                                  cmap=plt.cm.Blues,
#                                  normalize=normalize)
#     disp.ax_.set_title(title)
#
#     print(title)
#     print(disp.confusion_matrix)
#
# plt.show()


kernelMethod = ('linear', 'poly', 'rbf', 'sigmoid')
y_pos = np.arange(len(kernelMethod))
accuracy = result

plt.bar(y_pos, accuracy, align='center', alpha=0.5)
plt.xticks(y_pos, kernelMethod)
plt.ylabel('accuracy')
plt.title('NN accuracy vs Activation function')


plt.show()