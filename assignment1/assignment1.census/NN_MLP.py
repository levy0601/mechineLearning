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



X = data.iloc[:, 0:14]

y = data.iloc[:, 14]
y = np.array(y)
y = y.flatten()

scaler1 = preprocessing.StandardScaler()
scaler1.fit(X)
X = scaler1.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.55, random_state=42)

# print(train_X)
# print(train_Y)
# print(test_X)
# print(test_Y)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix


# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(X_train, y_train)
# y_pre = clf.predict(X_test)
# # pre_train_Y = clf.predict(train_X)
#
# print(clf.score(X_test, y_test))
# print(classification_report(y_test, y_pre, target_names = None))
# # print(classification_report(train_Y, pre_train_Y, target_names=None))
#
# np.set_printoptions(precision=2)
#
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

learningR = []

for learningRate in np.arange(0.001, 0.5, 0.01):
    clf = MLPClassifier(activation="tanh", solver='sgd', learning_rate_init=learningRate, alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train, y_train)
    y_pre = clf.predict(X_test)
    # pre_train_Y = clf.predict(train_X)

    print("learning rate " + str(learningRate) + " :" + str(clf.score(X_test, y_test)))
    learningR.append(clf.score(X_test, y_test))
    # print(classification_report(train_Y, pre_train_Y, target_names=None))

plt.figure()
plt.plot(np.arange(0.001, 0.5, 0.01), learningR, "#8B27CC")
plt.ylabel("Accuracy")
plt.xlabel("learningR")
plt.legend(['Test Data', 'Train Data'], loc=0, borderaxespad=0.2)
plt.text(np.max(learningR), np.argmax(learningR) * 0.01 + 0.001,
         ("x = " + str(np.max(learningR)) + "y = " + str(np.argmax(learningR) * 0.01 + 0.001)))
plt.show()

print("x = " + str(np.max(learningR)) + " y = " + str(np.argmax(learningR) * 0.01 + 0.001))
