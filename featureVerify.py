import numpy as np


def getPearsonCorrelationCoeff(feat, gt):
    return np.corrcoef(feat, gt)[1, 0]


def testForfeatures(features, gt, threshold):
    for i in range(features.shape[1]):
        print("Feature ", i + 1, "--> ", end="")
        coeff = getPearsonCorrelationCoeff(features[:, i], gt.T)
        if coeff > threshold:
            print("Great feature ==", coeff)
        elif 0.5 < coeff < 0.8:
            print("Good feature ==", coeff)
        elif 0 < coeff < 0.5:
            print("Ok feature ==", coeff)
        elif coeff < 0:
            print("Bad feature ==", coeff)