from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from numpy import mean
import numpy as np

dataset = np.loadtxt("data.txt", delimiter=(','))
x = dataset[:, list(range(15))]
y = dataset[:, -1]

Xtrain, Xtest, ytrain, y_test = train_test_split(x, y, test_size = 0.3, random_state =6, stratify=y)

c = [0.1, 0.01, 1, 0.0001, 0.0000001, 1.2, 1.3, 1.4]
max_c = c[0]; max_score = 0
for i in c:
    clf = LogisticRegression(C=i, solver="lbfgs", max_iter = 5000)
    sc = ((cross_val_score(clf, Xtrain, ytrain, cv=10)).mean())
    if max_score < sc:
        max_score = sc
        max_c = i
print("#########################################")
print("Max Score: {} | Max C: {}".format(max_score, max_c))
print("#########################################")
clf = LogisticRegression(C=max_c, solver="lbfgs", max_iter = 5000).fit(Xtrain, ytrain)
sc = ((cross_val_score(clf, Xtrain, ytrain, cv=10)).mean())

pY = clf.predict(Xtest)
#print(pY)
print("\n#################################################")
fs = f1_score(y_test, pY)
recall = recall_score(y_test, pY)
precision = precision_score(y_test, pY)
acc = accuracy_score(y_test, pY)
tn, fp, fn, tp = confusion_matrix(y_test, pY).ravel()
print("###################################################")
print("############  Regresión logística  ################")
print("###################################################")
print("##            Negative  Positive \t##")
print("##   Negative    {}        {}    \t##".format(tn, fp))
print("##   Positive    {}        {}    \t##".format(fn, tp))
print("###################################################")
print("###################################################")
print("\t   F1 Score: ",fs)
print("###################################################")
print("###################################################")
print("\t   Recall: ",recall)
print("###################################################")
print("###################################################")
print("\t   Precision: ", precision)
print("###################################################")
print("###################################################")
print("\t   Accuracy: ", acc)
