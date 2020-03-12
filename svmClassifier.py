from sklearn.svm import SVC as svm


class SVM:

    def __init__(self, feat, classes):
        self.model = svm(kernel='rbf', gamma=1e-4, C=1e+5)
        self.model.fit(feat, classes)

    def classify(self, testingData):
        return self.model.predict(testingData)