import numpy as np
from sklearn.neighbors import KDTree as kdt


class KdTree:

    def __init__(self, trainingData, k):
        self.tree = kdt(trainingData[:, :trainingData.shape[1]-1], k)
        self.trainingData = trainingData

    def classify(self, testingData):
        distance, index = self.tree.query(testingData)
        prediction = np.zeros(testingData.shape[0])
        for i in range(testingData.shape[0]):
            prediction[i] = max(set(self.trainingData[index[i]][:, self.trainingData.shape[1]-1]), key=list(self.trainingData[index[i]][:, self.trainingData.shape[1]-1]).count)
        return prediction