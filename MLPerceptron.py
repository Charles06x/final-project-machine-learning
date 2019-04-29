import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score

dataset = np.loadtxt("data.txt", delimiter=(','))
x = dataset[:, list(range(15))]
y = dataset[:, -1]

Xtrain, Xtest, ytrain, y_test = train_test_split(x, y, test_size = 0.3, random_state =6, stratify=y)
scaler = StandardScaler()
normalizedXTrain = scaler.fit_transform(Xtrain[:])

c = [0.001, 0.01, 1, 0.0001, 0.0000005, 1, 0.00000009, 0.0000000000004]
n = []
n = [x + 16 for x in range(len(c))]

max_c = c[0]; max_score = 0; max_n = n[0]
for i in range(len(c)):
    clf = MLPClassifier(hidden_layer_sizes=n[i], solver="lbfgs", alpha=c[i])
    sc = ((cross_val_score(clf, normalizedXTrain, ytrain, cv=10)).mean())
    if max_score < sc:
        max_score = sc
        max_c = c[i]
        max_n = n[i]

clasiffication = MLPClassifier(hidden_layer_sizes=max_n, solver="lbfgs", alpha=max_c)
cFitted = clasiffication.fit(Xtest, y_test)

pY = cFitted.predict(Xtest)
#print(pY)
print("\n#################################################")
fs = f1_score(y_test, pY)
recall = recall_score(y_test, pY)
precision = precision_score(y_test, pY)
acc = accuracy_score(y_test, pY)
tn, fp, fn, tp = confusion_matrix(y_test, pY).ravel()
print("###################################################")
print("##########  Multilayer Percerptron  ###############")
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
