import numpy as np
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn import datasets
from sklearn import svm

# Import Datos misteriosos data set
datos = np.loadtxt("datos.txt")
x = datos[:, 1:]
y = datos[:, 0]
y = y -1
n_features = x.shape[1]

targets = ["Clase1", "Clase2"]
n_clases = len(targets)

features = [str(i) for i in range(len(x[0]))]
n_features = len(x[0])

# Create output variables from original labels
output_y = np_utils.to_categorical(y)   # This is only required in

# Define MLP model
clf = Sequential()
clf.add(Dense(8, input_dim=n_features, activation='relu'))
clf.add(Dense(8, activation='relu'))
#clf.add(Dense(3, activation='softmax')) # for 2-class problems, use clf.add(Dense(1, activation='sigmoid'))
clf.add(Dense(1, activation='sigmoid'))
# Compile model
clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit model
clf.fit(x, output_y, epochs=150, batch_size=8)

# Predict class of a new observation
prob = clf.predict( datos[:, 1:] )
print("Probablities", prob)
print("Predicted class", np.argmax(prob, axis=-1))

# Evaluate model
kf = KFold(n_splits=5, shuffle = True)

acc = 0
recall = np.array([0., 0.])
for train_index, test_index in kf.split(x):
    # Training phase
    x_train = x[train_index, :]
    y_train = y[train_index]
    y_train = np_utils.to_categorical(y_train)

    clf = Sequential()
    clf.add(Dense(8, input_dim=n_features, activation='relu'))
    clf.add(Dense(8, activation='relu'))
    clf.add(Dense(3, activation='softmax'))
    #clf.compile(loss='categorical_crossentropy', optimizer='adam') # For 2-class problems, use clf.compile(loss='binary_crossentropy', optimizer='adam')
    clf.compile(loss='binary_crossentropy', optimizer='adam')
    clf.fit(x_train, y_train, epochs=150, batch_size=5, verbose=0)    

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]
    #y_pred = np.argmax(clf.predict(x_test), axis=-1)  # For 2-class problems, use (clf.predict(x_test) > 0.5).astype("int32")
    y_pred = np.argmax((clf.predict(x_test) > 0.5).astype("int32"))

    cm = confusion_matrix(y_test, y_pred)
    acc += (cm[0,0]+cm[1,1])/len(y_test)    

    recall[0] += cm[0,0]/(cm[0,0] + cm[0,1])
    recall[1] += cm[1,1]/(cm[1,0] + cm[1,1])

acc = acc/5
print('ACC = ', acc)

recall = recall/5
print('RECALL = ', recall)

# Train SVM classifier with all the available observations 
clf = svm.SVC(kernel = 'linear') 
clf.fit(x, y) 
 
# Predict one new sample 
print("Prediction for a new observation", clf.predict(datos[:, 1:]) ) 
 
# 5-fold cross-validation 
kf = KFold(n_splits=5, shuffle = True) 
clf = svm.SVC(kernel = 'linear') 
 
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
    acc_i = (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test)     
    print('acc = ', acc_i) 
    acc += acc_i  
 
acc = acc/5 
print('ACC = ', acc)