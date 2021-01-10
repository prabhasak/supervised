import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.utils import np_utils


class DataGet:

    def __init__(self,SamplesPath):
        self.path = SamplesPath

    def GetSamples(self):
        dataset = pd.read_csv(self.path,header = None)
        return dataset

    def PartitionData(self):
        data = self.GetSamples()
        data_y = data.iloc[1:,-1]
        data_x = data.iloc[1:,1:-1]
        X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3)
        return X_train, X_test, y_train, y_test

def main():
    InputPath = "eeg_data.csv"
    DataObj = DataGet(InputPath)
    X_train, X_test, y_train, y_test = DataObj.PartitionData()

    model=Sequential()
    model.add(Dense(64,activation='relu',input_dim=(178*1)))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(6,activation='softmax'))
    model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )
    y_train = np_utils.to_categorical(y_train)
    model.fit(X_train,y_train, epochs=10, batch_size=10)
    scores = model.evaluate(X_test,to_categorical(y_test),verbose=0)

    print("Accuracy is : ")
    print(scores[1]*100)

if __name__ == "__main__":
    main()