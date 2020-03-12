from bs4 import BeautifulSoup
import numpy as np
import csv
import sys
import kdTree as kd
import pickle as pk
from matplotlib import pyplot as plt
import svmClassifier as svm


class Digit:

    def __init__(self, ui, traces):
        self.ui = ui
        self.traces = traces


def ground_truth_dict(path):
    ground_truth = {}
    with open(path, mode='r') as infile:
        reader = csv.reader(infile, quotechar=None)
        ground_truth = {rows[0]: rows[1] for rows in reader}
    return ground_truth


def read_data(file, path):
    parsed = []
    with open(file) as f:
        for line in f:
            with open(path + '/' + line.strip()) as fp:
                soup = BeautifulSoup(fp, features='lxml')
                tag_ui = soup.find('annotation', {'type': 'UI'})
                traces = []
                tags = soup.find_all('trace')
                for trace in tags:
                    trace_split = trace.text.split(',')
                    trace_split = list(map(lambda x: x.split(), trace_split))
                    traces.append(np.array(trace_split, dtype=float))
                if tag_ui is not None:
                    ui = tag_ui.text
                    parsed.append(Digit(ui, traces))
    return parsed


def get_classes(training, gt):
    klass_to_int = {}
    int_to_klass = []
    for digit in training:
        symbol = gt[digit.ui]
        if symbol not in klass_to_int.keys():
            klass_to_int[symbol] = len(int_to_klass)
            int_to_klass.append(symbol)
    return klass_to_int, int_to_klass


def outputs(input, classification, classes, name):
    with open(name+'.txt', 'w') as o:
        for i in range(len(classification)):
            o.write(input[i].ui + ', ' + classes[int(classification[i])] + '\n')


def get_labels(data, ground_truth, classes):
    p = np.array(data)
    return np.vectorize(lambda x: classes[ground_truth[x.ui]])(p)


def normalize(coordinates):
    if len(coordinates) > 1:
        minValX = []
        minValY = []
        maxValX = []
        maxValY = []
        newCoordinates = []
        for stroke in coordinates:
            minValX.append(np.min(stroke[:, 0]))
            minValY.append(np.min(stroke[:, 1]))
            maxValX.append(np.max(stroke[:, 0]))
            maxValY.append(np.max(stroke[:, 1]))
        for stroke in coordinates:
            if np.max(maxValX) - np.min(minValX) == 0 and np.max(maxValY) - np.min(minValY) == 0:
                newCoordinates.append(np.column_stack((np.zeros(stroke.shape[0]), np.zeros(stroke.shape[0]))))
            elif np.max(maxValX) - np.min(minValX) == 0:
                newCoordinates.append(np.column_stack((np.zeros(stroke.shape[0]), ((2 * (stroke[:, 1] - np.min(minValY))) /
                                                        (np.max(maxValY) - np.min(minValY))) - 1)))
            elif np.max(maxValY) - np.min(minValY) == 0:
                newCoordinates.append(np.column_stack((((2 * (stroke[:, 0] - np.min(minValX))) /
                                                        (np.max(maxValX) - np.min(minValX))) - 1, np.zeros(stroke.shape[0]))))
            else:
                newCoordinates.append(np.column_stack((((2 * (stroke[:, 0] - np.min(minValX))) /
                                                        (np.max(maxValX) - np.min(minValX))) - 1,
                                                       ((2 * (stroke[:, 1] - np.min(minValY))) /
                                                           (np.max(maxValY) - np.min(minValY))) - 1)))
        return newCoordinates
    else:
        if np.max(coordinates[0][:, 0]) - np.min(coordinates[0][:, 0]) == 0 and np.max(coordinates[0][:, 1]) - np.min(coordinates[0][:, 1]) == 0:
            return [np.column_stack((np.zeros(coordinates[0].shape[0]), np.zeros(coordinates[0].shape[0])))]
        elif np.max(coordinates[0][:, 0]) - np.min(coordinates[0][:, 0]) == 0:
            return [np.column_stack((np.zeros(coordinates[0].shape[0]), ((2 * (coordinates[0][:, 1] - np.min(coordinates[0][:, 1]))) /
                                         (np.max(coordinates[0][:, 1]) - np.min(coordinates[0][:, 1]))) - 1))]
        elif np.max(coordinates[0][:, 1]) - np.min(coordinates[0][:, 1]) == 0:
            return [np.column_stack((((2 * (coordinates[0][:, 0] - np.min(coordinates[0][:, 0]))) /
                                      (np.max(coordinates[0][:, 0]) - np.min(coordinates[0][:, 0]))) - 1, np.zeros(coordinates[0].shape[0])))]
        else:
            return [np.column_stack((((2 * (coordinates[0][:, 0] - np.min(coordinates[0][:, 0]))) /
                                      (np.max(coordinates[0][:, 0]) - np.min(coordinates[0][:, 0]))) - 1,
                                 ((2 * (coordinates[0][:, 1] - np.min(coordinates[0][:, 1]))) /
                                  (np.max(coordinates[0][:, 1]) - np.min(coordinates[0][:, 1]))) - 1))]


def getlengthOfStroke(stroke):
    nextPoint = np.concatenate((stroke[1:, :], stroke[-1, :].reshape((1, stroke.shape[1]))))
    return np.sum(np.sqrt(np.sum(np.square(np.subtract(stroke, nextPoint)), axis=1, keepdims=True)))


def eliminateStrokes(data):
    for digit in data:
        strokes = digit.traces
        if len(strokes) > 1:
            newlist = []
            diagonalLength = getdiagLength(strokes)
            for stroke in strokes:
                length = getlengthOfStroke(stroke)
                if length > (0.1 * diagonalLength):
                    newlist.append(stroke)
            digit.traces = np.array(newlist)
            if digit.traces.shape[0] == 0:
                digit.traces = strokes
    return data


def getdiagLength(strokes):
    point = []
    minX = []
    minY = []
    maxX = []
    maxY = []
    for stroke in strokes:
        minX.append(np.min(stroke[:, 0]))
        minY.append(np.min(stroke[:, 1]))
        maxX.append(np.max(stroke[:, 0]))
        maxY.append(np.max(stroke[:, 1]))
    point.append([min(minX), min(minY)])
    point.append([max(maxX), max(maxY)])
    return np.sqrt(np.sum(np.square(np.subtract(point[0], point[1]))))


def line_length(strokes):
    total = 0
    for stroke in strokes:
        total += getlengthOfStroke(stroke)
    return total, total/len(strokes)


def combined_strokes(strokes):
    combined = strokes[0]
    for i in range(1, len(strokes)-1):
        combined = np.r_[combined, strokes[i]]
    return combined


def covariance_points(strokes):
    combined = combined_strokes(strokes)
    if combined.shape[0] == 1:
        return 0
    return np.cov(combined[:, 0], combined[:, 1])[0, 1]


def mean_x_y(strokes):
    combined = combined_strokes(strokes)
    return np.mean(combined[:, 0]), np.mean(combined[:, 1])


def slope(p1, p2):
    return (p2[1] - p1[1])/((p2[0] - p1[1])+1e-9)


def num_of_sharp_points(strokes):
    sharp_points = 0
    for stroke in strokes:
        sharp_points += 2
        a = []
        for i in range(len(stroke)-1):
            a.append(slope(stroke[i], stroke[i+1]))
        for i in range(1, len(a)-1):
            theta = a[i] - a[i+1]
            theta2 = a[i-1] - a[i]
            if theta == 0:
                continue
            delta = theta * theta2
            if (delta <= 0) and (theta2 != 0):
                sharp_points += 1
    total_points = (combined_strokes(strokes)).shape[0]
    return sharp_points, sharp_points/total_points


def orientation(p1, p2, p3):
    val = (p2[1] - p1[1]) * (p3[0] - p2[0]) - (p2[0] - p1[0]) * (p3[1] - p2[1])
    if val > 0:
        return 1
    elif val < 0:
        return 2
    else:
        return 0


def onSegment(p1, p2, p3):
    return max(p1[0], p3[0]) >= p2[0] >= min(p1[0], p3[0]) and \
           max(p1[1], p3[1]) >= p2[1] >= min(p1[1], p3[1])


def line_intersect(p1, p2, q1, q2):
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if (o1 != o2) and (o3 != o4):
        return True

    if (o1 == 0) and onSegment(p1, q1, p2):
        return True

    if (o2 == 0) and onSegment(p1, q2, p2):
        return True

    if (o3 == 0) and onSegment(q1, p1, q2):
        return True

    if (o4 == 0) and onSegment(q1, p2, q2):
        return True

    return False


def horizontal_vertical_lines(minV, maxV):
    ys = np.linspace(minV, maxV, 6)
    xs = np.array([minV, maxV])
    horizontal = []
    vertical = []
    for i in range(ys.shape[0]-1):
        gy = np.linspace(ys[i]+.05, ys[i+1]-.05, 9)
        x, y = np.meshgrid(xs, gy)
        horizontal.append([np.c_[x[:, 0], y[:, 0]], np.c_[x[:, 1], y[:, 1]]])
        vertical.append([np.c_[y[:, 0], x[:, 0]], np.c_[y[:, 1], x[:, 1]]])
    return np.array(horizontal),  np.array(vertical)


def plot_traces(stokes):
    for stroke in stokes:
        plt.plot(stroke[:, 0], stroke[:, 1])
    plt.ylim((-2, 2))
    plt.xlim((-2, 2))
    plt.show()


def crossing_features(strokes):
    h, v = horizontal_vertical_lines(-1, 1)
    h_total = []
    v_total = []
    # plot_traces(strokes)
    for hgroup, vgroup in zip(h[:], v[:]):
        htotal_cross = 0
        vtotal_cross = 0
        for i in range(9):
            hp1 = hgroup[0, i]
            hp2 = hgroup[1, i]
            vp1 = vgroup[0, i]
            vp2 = vgroup[1, i]
            for stroke in strokes:
                for j in range(len(stroke)-1):
                    if line_intersect(hp1, hp2, stroke[j], stroke[j+1]):
                        htotal_cross += 1
                    if line_intersect(vp1, vp2, stroke[j], stroke[j+1]):
                        vtotal_cross += 1
        h_total.append(htotal_cross/9)
        v_total.append(vtotal_cross/9)
    return h_total, v_total


def aspectRatio(coordinates):
    minX = []
    minY = []
    maxX = []
    maxY = []
    for stroke in coordinates:
        minX.append(np.min(stroke[:, 0]))
        minY.append(np.min(stroke[:, 1]))
        maxX.append(np.max(stroke[:, 0]))
        maxY.append(np.max(stroke[:, 1]))
    width = max(maxX) - min(minX)
    height = max(maxY) - min(minY)
    return height, width


def getNumberOfStroke(coordinates):
    return len(coordinates)


def fuzzyHist(coordinates):
    points = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
    corners = np.column_stack((points[0].flatten(), points[1].flatten()))
    symbcoord = []
    for strokes in coordinates:
        for stroke in strokes:
            symbcoord.append(stroke)
    p1 = np.array(symbcoord)
    p2 = np.concatenate((p1[1:, :], p1[-1, :].reshape((1, p1.shape[1]))))
    horizVec = [2, 0]
    verticalVec = [0, -2]
    diag1Vec = [2, 2]
    diag2Vec = [2, -2]
    horizAxisLength = 2
    verticalAxisLength = 2
    diag1Length = 2.828
    diag2Length = 2.828
    bins = []
    for p3 in corners:
        cornerBins = []
        dist = np.linalg.norm(np.cross(p2[:-1] - p1[:-1], p1[:-1] - p3)) / ((np.linalg.norm(p2[:-1] - p1[:-1], axis=1)) + 1e-9)
        if dist.size == 0:
            cornerBins.append(np.array([0, 0, 0, 0]))
            bins.append(cornerBins)
            continue
        indices = dist.argsort()
        p1sorted = p1[indices]
        p2sorted = p2[indices]
        closestPoint1, closestPoint2 = p1sorted[:1, :], p2sorted[:1, :]
        lineVect = [closestPoint2[0][0] - closestPoint1[0][0], closestPoint2[0][1] - closestPoint1[0][1]]
        linelength = np.sqrt(np.square(lineVect[0]) + np.square(lineVect[1])) + 1e-9
        for vec, length in zip([horizVec, verticalVec, diag1Vec, diag2Vec], [horizAxisLength, verticalAxisLength, diag1Length, diag2Length]):
            if np.dot(lineVect, vec) / (linelength * length) < -1:
                ans = -1
            elif np.dot(lineVect, vec) / (linelength * length) > 1:
                ans = 1
            else:
                ans = np.dot(lineVect, vec) / (linelength * length)
            val = np.arccos(ans)
            cornerBins.append(val)
        bins.append(cornerBins)
    return np.array(bins).flatten()


def pre_processing_data(data, name):
    for digit in data:
        trace = digit.traces
        digit.traces = normalize(trace)
    eliminated = eliminateStrokes(data)
    train = open(name, 'wb')
    pk.dump(eliminated, train)
    train.close()
    load = open(name, 'rb')
    eliminated = pk.load(load)
    FeatureVector = []
    for digit in eliminated:
        perSampleFeature = []
        perSampleFeature.append(covariance_points(digit.traces))
        total, average = line_length(digit.traces)
        perSampleFeature.append(total)
        perSampleFeature.append(average)
        x, y = mean_x_y(digit.traces)
        perSampleFeature.append(x)
        perSampleFeature.append(y)
        total_sharp, average_sharp = num_of_sharp_points(digit.traces)
        perSampleFeature.append(total_sharp)
        perSampleFeature.append(average_sharp)
        height, width = aspectRatio(digit.traces)
        perSampleFeature.append(height)
        perSampleFeature.append(width)
        perSampleFeature.append(getNumberOfStroke(digit.traces))
        val1, val2 = crossing_features(digit.traces)
        for v1 in val1:
            perSampleFeature.append(v1)
        for v2 in val2:
            perSampleFeature.append(v2)
        hist = fuzzyHist(digit.traces)
        for h in hist:
            perSampleFeature.append(h)
        FeatureVector.append(perSampleFeature)
    return FeatureVector


def training_pipeline(featureVectors, labels, classifier, classifier2):
    """
    Pipe line for handling training a model
    :param training: Array of training data already read in.
    :param ground_truth: Dictionary of UI to the ground truth classification
    :param pre_processing: Function pointer to function that will provide the pre-processing.
    :param classifier: Function to generate the model from the preprocessed data.
    :return: A model that has been trained on the training data and can be used to classify
    """
    print("Training vector ")
    modelkD = classifier(np.c_[featureVectors, labels])
    md = open('kdTreeModel.txt', 'wb')
    pk.dump(modelkD, md)
    md.close()

    modelSvm = classifier2(featureVectors, labels)
    md = open('svmModel.txt', 'wb')
    pk.dump(modelSvm, md)
    md.close()

    return modelkD, modelSvm


def classify_pipeline(featureVectors, model):
    """
    Pipeline for handling the running of a trained model on input
    :param input_values: Array of data to classify
    :param pre_processing: function used to preprocess data for the format expected by model
    :param model: trained model to use on the data
    :return: matrix which possible labels for each input
    """
    return model.classify(np.array(featureVectors))


def pre_processing_pipeline():
    """
    Pipeline for reading the training, junk and test data set, pre-process and generate feature vectors.
    :param system arguments
    :return: training and test feature vectors
    """
    if int(sys.argv[1]) == 1:

        ground_truth = ground_truth_dict(sys.argv[4])

        #####################     read training data and create feature vector ###########################
        training = read_data(sys.argv[2], sys.argv[3])
        train = open('data.txt', 'wb')
        pk.dump(training, train)
        train.close()
        training_classes, training_int_classes = get_classes(training, ground_truth)
        training_labels = get_labels(training, ground_truth, training_classes)
        trainingFeatureVectors = pre_processing_data(training, "procdata.txt")
        fVec = open('featureVec.txt', 'wb')
        pk.dump(trainingFeatureVectors, fVec)
        fVec.close()

        #####################     read testing data and create feature vector #############################
        testing = read_data(sys.argv[5], sys.argv[3])
        test = open('testdata.txt', 'wb')
        pk.dump(testing, test)
        test.close()
        testing_classes, testing_int_classes = get_classes(training, ground_truth)
        testing_labels = get_labels(testing, ground_truth, testing_classes)
        testingFeatureVectors = pre_processing_data(testing, 'TestProcdata.txt')
        fVec = open('TestfeatureVec.txt', 'wb')
        pk.dump(testingFeatureVectors, fVec)
        fVec.close()

        junk_ground_truth = ground_truth_dict(sys.argv[8])

        ######################    read training junk and create junk feature vector #######################
        training_Junk = read_data(sys.argv[6], sys.argv[7])
        trainJunk = open('trainJunkdata.txt', 'wb')
        pk.dump(training_Junk, trainJunk)
        trainJunk.close()
        junk_classes, junk_int_classes = get_classes(training_Junk, junk_ground_truth)
        for key in junk_classes:
            junk_classes[key] = len(training_classes)
        junk_labels = get_labels(training_Junk, junk_ground_truth, junk_classes)
        trainJunkFeatureVectors = pre_processing_data(training_Junk, 'TrainJunkProcdata.txt')
        tJfVec = open('TrainJunkfeatureVec.txt', 'wb')
        pk.dump(trainJunkFeatureVectors, tJfVec)
        tJfVec.close()

        ######################    read testing junk and create junk feature vector #######################
        testing_junk = read_data(sys.argv[9], sys.argv[7])
        testJ = open('testJunkdata.txt', 'wb')
        pk.dump(testing_junk, testJ)
        testJ.close()
        junkTest_classes, junkTest_int_classes = get_classes(testing_junk, junk_ground_truth)
        for key in junkTest_classes:
            junkTest_classes[key] = len(training_classes)
        junkTest_labels = get_labels(testing_junk, junk_ground_truth, junkTest_classes)
        testJunkFeatureVectors = pre_processing_data(testing_junk, 'TestJunkProcdata.txt')
        tJfVec = open('TestJunkfeatureVec.txt', 'wb')
        pk.dump(testJunkFeatureVectors, tJfVec)
        tJfVec.close()

        ############################################  merge sample with junk ################################################

        trainingFeatureVectors = np.r_[trainingFeatureVectors, trainJunkFeatureVectors]
        testingFeatureVectors = np.r_[testingFeatureVectors, testJunkFeatureVectors]

        training_labels = np.r_[training_labels, junk_labels]
        testing_labels = np.r_[testing_labels, junkTest_labels]

        training_int_classes = np.r_[training_int_classes, junk_int_classes]
        testing_int_classes = np.r_[testing_int_classes, junkTest_int_classes]

        training = np.r_[training, training_Junk]
        testing = np.r_[testing, testing_junk]

        return trainingFeatureVectors, testingFeatureVectors, training_labels, testing_labels, training_int_classes, testing_int_classes, training, testing
    elif int(sys.argv[1]) == 2:

        # read processed training data and featureVectors
        load = open('procdata.txt', 'rb')
        training = pk.load(load)
        ground_truth = ground_truth_dict(sys.argv[4])
        training_classes, training_int_classes = get_classes(training, ground_truth)
        training_labels = get_labels(training, ground_truth, training_classes)
        load = open('featureVec.txt', 'rb')
        trainingFeatureVectors = pk.load(load)

        # read new testing data with new ground truth files

        testing = read_data(sys.argv[5], sys.argv[3])
        test = open('testdata.txt', 'wb')
        pk.dump(testing, test)
        test.close()
        testing_classes, testing_int_classes = get_classes(training, ground_truth)
        testing_labels = get_labels(testing, ground_truth, testing_classes)
        testingFeatureVectors = pre_processing_data(testing, 'TestProcdata.txt')
        fVec = open('TestfeatureVec.txt', 'wb')
        pk.dump(testingFeatureVectors, fVec)
        fVec.close()

        # read processed junk data and featureVectors
        load = open('TrainJunkProcdata.txt', 'rb')
        training_Junk = pk.load(load)
        junk_ground_truth = ground_truth_dict(sys.argv[8])
        junk_classes, junk_int_classes = get_classes(training_Junk, junk_ground_truth)
        for key in junk_classes:
            junk_classes[key] = len(training_classes)
        junk_labels = get_labels(training_Junk, junk_ground_truth, junk_classes)
        load = open('TrainJunkfeatureVec.txt', 'rb')
        trainJunkFeatureVectors = pk.load(load)

        # read new junk testing data and new ground truth files
        testing_junk = read_data(sys.argv[9], sys.argv[7])
        testJ = open('testJunkdata.txt', 'wb')
        pk.dump(testing_junk, testJ)
        testJ.close()
        junkTest_classes, junkTest_int_classes = get_classes(testing_junk, junk_ground_truth)
        for key in junkTest_classes:
            junkTest_classes[key] = len(training_classes)
        junkTest_labels = get_labels(testing_junk, junk_ground_truth, junkTest_classes)
        testJunkFeatureVectors = pre_processing_data(testing_junk, 'TestJunkProcdata.txt')
        tJfVec = open('TestJunkfeatureVec.txt', 'wb')
        pk.dump(testJunkFeatureVectors, tJfVec)
        tJfVec.close()

        ############################################  merge sample with junk ################################################

        trainingFeatureVectors = np.r_[trainingFeatureVectors, trainJunkFeatureVectors]
        testingFeatureVectors = np.r_[testingFeatureVectors, testJunkFeatureVectors]

        training_labels = np.r_[training_labels, junk_labels]
        testing_labels = np.r_[testing_labels, junkTest_labels]

        training_int_classes = np.r_[training_int_classes, junk_int_classes]
        testing_int_classes = np.r_[testing_int_classes, junkTest_int_classes]

        training = np.r_[training, training_Junk]
        testing = np.r_[testing, testing_junk]

        return trainingFeatureVectors, testingFeatureVectors, training_labels, testing_labels, training_int_classes, testing_int_classes, training, testing
    elif int(sys.argv[1]) == 3:

        ground_truth = ground_truth_dict(sys.argv[4])

        # read training feature vector
        load = open('procdata.txt', 'rb')
        training = pk.load(load)
        training_classes, training_int_classes = get_classes(training, ground_truth)
        training_labels = get_labels(training, ground_truth, training_classes)
        load = open('featureVec.txt', 'rb')
        trainingFeatureVectors = pk.load(load)

        # read testing feature vector
        load = open('TestProcdata.txt', 'rb')
        testing = pk.load(load)
        testing_classes, testing_int_classes = get_classes(training, ground_truth)
        testing_labels = get_labels(testing, ground_truth, testing_classes)
        load = open('TestfeatureVec.txt', 'rb')
        testingFeatureVectors = pk.load(load)

        # read training junk feature vector
        load = open('TrainJunkProcdata.txt', 'rb')
        training_Junk = pk.load(load)
        junk_ground_truth = ground_truth_dict(sys.argv[8])
        junk_classes, junk_int_classes = get_classes(training_Junk, junk_ground_truth)
        for key in junk_classes:
            junk_classes[key] = len(training_classes)
        junk_labels = get_labels(training_Junk, junk_ground_truth, junk_classes)
        load = open('TrainJunkfeatureVec.txt', 'rb')
        trainJunkFeatureVectors = pk.load(load)

        # read testing junk feature vector
        load = open('TestJunkProcdata.txt', 'rb')
        testing_junk = pk.load(load)
        junkTest_classes, junkTest_int_classes = get_classes(testing_junk, junk_ground_truth)
        for key in junkTest_classes:
            junkTest_classes[key] = len(training_classes)
        junkTest_labels = get_labels(testing_junk, junk_ground_truth, junkTest_classes)
        load = open('TestJunkfeatureVec.txt', 'rb')
        testJunkFeatureVectors = pk.load(load)

        trainingFeatureVectors = np.r_[trainingFeatureVectors, trainJunkFeatureVectors]
        testingFeatureVectors = np.r_[testingFeatureVectors, testJunkFeatureVectors]

        training_labels = np.r_[training_labels, junk_labels]
        testing_labels = np.r_[testing_labels, junkTest_labels]

        training_int_classes = np.r_[training_int_classes, junk_int_classes]
        testing_int_classes = np.r_[testing_int_classes, junkTest_int_classes]

        training = np.r_[training, training_Junk]
        testing = np.r_[testing, testing_junk]

        return trainingFeatureVectors, testingFeatureVectors, training_labels, testing_labels, training_int_classes, testing_int_classes, training, testing


def acc(pred, gt):
    print((1 - np.count_nonzero(np.subtract(pred, gt))/gt.shape[0]) * 100, "% accuracy")


if __name__ == '__main__':
    if len(sys.argv) < 10:
        print('Usage: [[Python]] digitClassifer.py [1] [training file] [path to training data] [ground truth file] '
              '[testing file] [training junk file] [path to junk data] [ground truth file] [testing junk file]')
    else:
        trainVec, testVec, train_labels, test_labels, train_int_classes, test_int_classes, training, testing = pre_processing_pipeline()
        if int(sys.argv[1]) == 1:
            model, model2 = training_pipeline(trainVec, train_labels, lambda x: kd.KdTree(x, 1), lambda x1, x2: svm.SVM(x1, x2))
        else:
            load = open('kdTreeModel.txt', 'rb')
            model = pk.load(load)
            load = open('svmModel.txt', 'rb')
            model2 = pk.load(load)

        # kdtree training and testing results
        print("\nTesting KDTREE->")
        kdtest_classified = classify_pipeline(trainVec, model)
        print("\nTraining Accuracy:", end="")
        acc(kdtest_classified, train_labels)
        outputs(training, kdtest_classified, train_int_classes, 'kdTree-training')

        kdtest_classified = classify_pipeline(testVec, model)
        print("\nTesting Accuracy:", end="")
        acc(kdtest_classified, test_labels)
        outputs(testing, kdtest_classified, test_int_classes, 'kdTree-testing')

        # svm training and testing results
        print("\nTesting SVM->")
        svmtest_classified = classify_pipeline(trainVec, model2)
        print("\nTraining Accuracy:", end="")
        acc(svmtest_classified, train_labels)
        outputs(training, svmtest_classified, train_int_classes, 'svm-training')

        svmtest_classified = classify_pipeline(testVec, model2)
        print("\nTesting Accuracy:", end="")
        acc(svmtest_classified, test_labels)
        outputs(testing, svmtest_classified, test_int_classes, 'svm-testing')
