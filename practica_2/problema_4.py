"""
Problema 4
Calcula la exactitud de un clasificador k-NN
con k-fold cross validation (5 pliegues)
para k = 1, 2, 3, 4, ..., 10 del clasificador.
"""
import matplotlib.pyplot as plt
import numpy as np

from glob import glob
from sklearn import svm
from random import random
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier



# Import datos misteriosos
files = glob("*.txt")
with open(files[0], "r") as f:
    data = [
        register.split()[1:]
        for register
        in f.readlines()
    ]

print(len(data))
# Train KNN classifier with all the available observations
sample_size = 3*len(data)//4
x = data[sample_size:]
y = data[:sample_size]
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(x,y)

# Predict one new sample
# print("Prediction for a new observation", clf.predict( [[random() * 2 for _ in range(len(data[0]))]] ))

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle = True)
acc = 0
for train_index, test_index in kf.split(x):
    # Training phase
    x_train = x[train_index, :]
    y_train = y[train_index]
    clf.fit(x_train, y_train)

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]    
    y_pred = clf.predict(x_test)

    # Calculate confusion matrix and model performance
    cm = confusion_matrix(y_test, y_pred)
    acc_i = (sum(cm[i,i] for i in range(len(cm))))/len(y_test)    
    print('acc = ', acc_i)

    acc += acc_i 

acc = acc/5
print('ACC = ', acc)