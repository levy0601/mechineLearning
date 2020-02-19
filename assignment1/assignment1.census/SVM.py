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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# print(train_X)
# print(train_Y)
# print(test_X)
# print(test_Y)

from sklearn import svm, metrics
from sklearn.metrics import classification_report, plot_confusion_matrix

#
# result = []
# kernel = "linear"
# clf = svm.SVC(kernel=kernel,decision_function_shape='ovo')
# clf.fit(X_train, y_train)
# y_pre_ovo = clf.predict(X_test)
# result.append(clf.score(X_test, y_test))
# # pre_train_Y = clf.predict(train_X)
#
#
# print(kernel + " " + str( clf.score(X_test, y_test)))
#
# kernel = "poly"
# clf = svm.SVC(kernel=kernel,decision_function_shape='ovo')
# clf.fit(X_train, y_train)
# y_pre_ovo = clf.predict(X_test)
# result.append(clf.score(X_test, y_test))
# # pre_train_Y = clf.predict(train_X)
#
# print(kernel + " " + str( clf.score(X_test, y_test)))
#
#
# kernel = "rbf"
# clf = svm.SVC(kernel=kernel,decision_function_shape='ovo')
# clf.fit(X_train, y_train)
# y_pre_ovo = clf.predict(X_test)
# result.append(clf.score(X_test, y_test))
# # pre_train_Y = clf.predict(train_X)
#
# print(kernel + " " + str( clf.score(X_test, y_test)))
#
#
# kernel = "sigmoid"
# clf = svm.SVC(kernel=kernel,decision_function_shape='ovo')
# clf.fit(X_train, y_train)
# y_pre_ovo = clf.predict(X_test)
# result.append(clf.score(X_test, y_test))
# # pre_train_Y = clf.predict(train_X)
#
# print(kernel + " " + str( clf.score(X_test, y_test)))



# print(classification_report(y_test, y_pre_ovo, target_names = None))
# print(classification_report(train_Y, pre_train_Y, target_names=None))

# np.set_printoptions(precision=2)

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


kernelMethod = ('poly', 'rbf', 'sigmoid')
# y_pos = np.arange(len(kernelMethod))
# accuracy = result
#
# plt.bar(y_pos, accuracy, align='center', alpha=0.5)
# plt.xticks(y_pos, kernelMethod)
# plt.ylabel('accuracy')
# plt.title('Census SVM accuracy in Different kernel : census')
#

# plt.show()


gamma = 0.1
for kernel in kernelMethod:
    test = []
    train = []
    for gamma in np.arange(0.00001, 1, 0.1):
        clf = svm.SVC(kernel=kernel,decision_function_shape='ovo',gamma=gamma)
        clf.fit(X_train, y_train)
        y_pre_ovo = clf.predict(X_test)
        test.append(clf.score(X_test, y_test))
        #train.append(clf.score(X_train,y_train))
        print(gamma)

    print(kernel)
    plt.figure()
    plt.plot(np.arange(0, 1, 0.1),test,"#8B27CC")
    #plt.plot(np.arange(0, 1, 0.05),train,"#25B71A")
    plt.text(np.argmax(test) * 0.1 , np.max(test),
             ("x = " + str(np.argmax(test) * 0.1 )) + " y = " + str(np.max(test)))
    plt.ylabel("Accuracy")
    plt.xlabel("gamma")
    plt.legend(['Test Data', 'Train Data'],  loc=0, borderaxespad=0.2)
    plt.title("SVM relation between Accuracy vs Gamma : census," + kernel)
    plt.show()