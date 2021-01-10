import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.utils import np_utils
import scipy.io
import h5py

import numpy as np
import h5py

#from sklearn.svm import SVC,LinearSVC
#from sklearn.model_selection import train_test_split
##clf = SVC(kernel='sigmoid',gamma='auto')
#clf = LinearSVC(random_state=0, tol=1e-5)
#X_train, X_test, y_train, y_test = train_test_split(board_data[:,0:2],board_data[:,2], test_size=0.33, random_state=42)
#clf.fit(X_train,y_train)
#preds = clf.predict(X_test)
#scores = clf.score(X_test,y_test)

def main():
    num_classes = 10

    X_train =  scipy.io.loadmat('../assignment-4-MNIST/MNIST/mnist-rot_training_data.mat')
    Y_train = scipy.io.loadmat('../assignment-4-MNIST/MNIST/mnist-rot_training_label.mat')
    X_test = scipy.io.loadmat('../assignment-4-MNIST/MNIST/mnist-rot_test_data.mat')
    Y_test = scipy.io.loadmat('../assignment-4-MNIST/MNIST/mnist-rot_test_label.mat')
    X_train = X_train['train_data']
    Y_train = Y_train['train_label']
    X_test = X_test['test_data']
    Y_test = Y_test['test_label']

    #x_train = np.array(list(X_train.items()), dtype=dtype)


    X_train = X_train.reshape(12000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    Y_test = keras.utils.to_categorical(Y_test, num_classes)

    Y_train = Y_train.reshape(12000,10)
    Y_test = Y_test.reshape(10000,10)

    model=Sequential()
    model.add(Dense(64,activation='relu',input_dim=(28*28)))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    from keras.optimizers import RMSprop,Adam,SGD
    model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam', metrics = ['accuracy'] )

    model.fit(X_train,Y_train, epochs=10, batch_size=64)
    scores = model.evaluate(X_test,Y_test,verbose=0)

    print("Accuracy is : ")
    print(scores[1]*100)

if __name__ == "__main__":
    main()
