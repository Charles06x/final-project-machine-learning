from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from numpy import mean
import numpy as np

dataset = np.loadtxt("data.txt", delimiter=(','))
x = dataset[:, list(range(15))]
y = dataset[:, -1]

Xtrain, Xtest, ytrain, y_test = train_test_split(x, y, test_size = 0.3, random_state =6, stratify=y)
scaler = StandardScaler()
normalizedXTrain = scaler.fit_transform(Xtrain[:])

c = [1, 1.2, 1.3, 1.4, 5, 20, 15, 40, 100]
max_c = c[0]; max_score = 0
for i in c:
    svm = SVC(C=i, gamma="scale")
    sc = ((cross_val_score(svm, normalizedXTrain, ytrain, cv=10)).mean())
    if max_score < sc:
        max_score = sc
        max_c = i
print("#########################################")
print("Max Score | Max_C")
print("{} | {}".format(max_score, max_c))
print("#########################################")
svm = SVC(C=i, gamma="scale")
sc = ((cross_val_score(svm, normalizedXTrain, ytrain, cv=10)).mean())
scFitted = svm.fit(Xtrain, ytrain)
pY = scFitted.predict(Xtest)
#print(pY)
print("\n#########################################")
fs = f1_score(y_test, pY)
recall = recall_score(y_test, pY)
precision = precision_score(y_test, pY)
acc = accuracy_score(y_test, pY)
tn, fp, fn, tp = confusion_matrix(y_test, pY).ravel()
print("\tF1 Score: ",fs)
print("###################################################")
print("###################################################")
print("\tRecall: ",recall)
print("###################################################")
print("###################################################")
print("\tPrecision: ", precision)
print("###################################################")
print("###################################################")
print("\tAccuracy: ", acc)
print("###################################################")
print("###################################################")
print("##            Negative  Positive \t##".format(tn, fp))
print("##   Negative    {}        {}    \t##".format(tn, fp))
print("##   Positive    {}        {}    \t##".format(fn, tp))
print("###################################################")
print("###################################################")
