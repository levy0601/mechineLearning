import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# encoder_x =LabelEncoder()

data = pd.read_csv("../../data set/Car Evaluation/car.csv")
# data.iloc[:,9] = encoder_x.fit_transform(data.iloc[:,9])
# print(data)

X = data.iloc[:, 0:6]

y = data.iloc[:, 6]
y = np.array(y)
y = y.flatten()

scaler1 = preprocessing.StandardScaler()
scaler1.fit(X)
X = scaler1.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# print(train_X)
# print(train_Y)
# print(test_X)
# print(test_Y)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix

# result = [];
#
# clf = MLPClassifier(activation="identity",solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(X_train, y_train)
# y_pre = clf.predict(X_test)
# # pre_train_Y = clf.predict(train_X)
#
# print(clf.score(X_test, y_test))
# print(classification_report(y_test, y_pre, target_names = None))
# # print(classification_report(train_Y, pre_train_Y, target_names=None))
#
# result.append(clf.score(X_test, y_test))
# np.set_printoptions(precision=2)
#
#
# clf = MLPClassifier(activation="logistic",solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(X_train, y_train)
# y_pre = clf.predict(X_test)
# # pre_train_Y = clf.predict(train_X)
#
# print(clf.score(X_test, y_test))
# print(classification_report(y_test, y_pre, target_names = None))
# # print(classification_report(train_Y, pre_train_Y, target_names=None))
# result.append(clf.score(X_test, y_test))
# np.set_printoptions(precision=2)
#
#
# clf = MLPClassifier(activation= "relu",solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(X_train, y_train)
# y_pre = clf.predict(X_test)
# # pre_train_Y = clf.predict(train_X)
#
# print(clf.score(X_test, y_test))
# print(classification_report(y_test, y_pre, target_names = None))
# # print(classification_report(train_Y, pre_train_Y, target_names=None))
# result.append(clf.score(X_test, y_test))
# np.set_printoptions(precision=2)
#
#
# clf = MLPClassifier(activation= "tanh",solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(X_train, y_train)
# y_pre = clf.predict(X_test)
# # pre_train_Y = clf.predict(train_X)
#
# print(clf.score(X_test, y_test))
# print(classification_report(y_test, y_pre, target_names = None))
# # print(classification_report(train_Y, pre_train_Y, target_names=None))
# result.append(clf.score(X_test, y_test))
# np.set_printoptions(precision=2)
#
#
#
# activationMethod = ('identity', 'logistic', 'relu', 'tanh')
# y_pos = np.arange(len(activationMethod))
# accuracy = result
#
# plt.bar(y_pos, accuracy, align='center', alpha=0.5)
# plt.xticks(y_pos, activationMethod)
# plt.ylabel('accuracy')
# plt.title('NN accuracy vs Activation function')
#
# plt.show()


# learning rate

# learningRate = 0.001
#
# clf = MLPClassifier(activation= "tanh",solver='sgd', learning_rate_init =learningRate ,alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(X_train, y_train)
# y_pre = clf.predict(X_test)
# # pre_train_Y = clf.predict(train_X)
#
# print("learning rate " + str(learningRate) + " :" + str(clf.score(X_test, y_test)))
# # print(classification_report(train_Y, pre_train_Y, target_names=None))

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
plt.text(np.argmax(learningR), np.argmax(learningR) * 0.01 + 0.001,
         ("x = " + str(np.max(learningR)) + "y = " + str(np.argmax(learningR) * 0.01 + 0.001)))
plt.ylabel("Accuracy")
plt.xlabel("learningR")
plt.legend(['Test Data', 'Train Data'], loc=0, borderaxespad=0.2)
plt.show()

print("x = " + str(np.max(learningR)) + " y = " + str(np.argmax(learningR) * 0.01 + 0.001))
