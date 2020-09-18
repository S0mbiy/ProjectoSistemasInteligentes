#------------------------------------------------------------------------------------------------------------------
import numpy as np

from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

# Import IRIS data set
iris = datasets.load_iris()
x = iris.data
y = iris.target

targets = iris.target_names
n_clases = len(targets)

features = iris.feature_names
n_features = len(features)

# Create output variables from original labels
output_y = np_utils.to_categorical(y)   # This is only required in 
print(output_y)

# Define MLP model
clf = Sequential()
clf.add(Dense(8, input_dim=n_features, activation='relu'))
clf.add(Dense(8, activation='relu'))
clf.add(Dense(3, activation='softmax')) # for 2-class problems, use clf.add(Dense(1, activation='sigmoid'))

# Compile model
clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
# Fit model
clf.fit(x, output_y, epochs=150, batch_size=5)

# Predict class of a new observation
prob = clf.predict( [[1.,2.,3.,4.]] )
print("Probablities", prob)
print("Predicted class", np.argmax(prob, axis=-1))

# Evaluate model
kf = KFold(n_splits=5, shuffle = True)

acc = 0
recall = np.array([0., 0., 0.])
for train_index, test_index in kf.split(x):

    # Training phase
    x_train = x[train_index, :]
    y_train = y[train_index]
    y_train = np_utils.to_categorical(y_train)

    clf = Sequential()
    clf.add(Dense(8, input_dim=n_features, activation='relu'))
    clf.add(Dense(8, activation='relu'))
    clf.add(Dense(3, activation='softmax'))
    clf.compile(loss='categorical_crossentropy', optimizer='adam') # For 2-class problems, use clf.compile(loss='binary_crossentropy', optimizer='adam')
    clf.fit(x_train, y_train, epochs=150, batch_size=5, verbose=0)    

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]
    y_pred = np.argmax(clf.predict(x_test), axis=-1)  # For 2-class problems, use (clf.predict(x_test) > 0.5).astype("int32")

    cm = confusion_matrix(y_test, y_pred)

    acc += (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test)    

    recall[0] += cm[0,0]/(cm[0,0] + cm[0,1] + cm[0,2])
    recall[1] += cm[1,1]/(cm[1,0] + cm[1,1] + cm[1,2])
    recall[2] += cm[2,2]/(cm[2,0] + cm[2,1] + cm[2,2])


acc = acc/5
print('ACC = ', acc)

recall = recall/5
print('RECALL = ', recall)