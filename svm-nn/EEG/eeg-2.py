from sklearn.svm import SVC,LinearSVC
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

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
        X_train = (np.array(X_train))
        X_test = (np.array(X_test))
        y_train = (np.array(y_train))
        y_test = (np.array(y_test))
        return X_train.astype(int), X_test.astype(int), y_train.astype(int), y_test.astype(int)

def main():
    InputPath = "eeg_data.csv"
    DataObj = DataGet(InputPath)
    X_train, X_test, y_train, y_test = DataObj.PartitionData()

    clf = LinearSVC(random_state=0, tol=1e-5)
    clf1 = SVC(kernel='rbf',gamma='auto')
    clf.fit(X_train,y_train)
    clf1.fit(X_train,y_train)
    preds = clf.predict(X_test)
    pred1 = clf1.predict(X_test)
    scores = clf.score(X_test,y_test)
    scores1 = clf1.score(X_test,y_test)
    print(scores)
    print(scores1)

if __name__ == "__main__":
    main()