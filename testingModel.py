class testModel:

    def __init__(self, training, labels):
        self.training = training
        self.labels = labels
        self.klass = 0

    def classify(self, input):
        return self.klass
