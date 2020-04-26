import features
import numpy as np
import scipy.optimize as opt
import time

"""This is the activation function"""
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(y):
    """
    derivative of sigmoid
    in this function y is already sigmoided
    """
    return y * (1.0 - y)


class NeuralNetworkClassifier():
    def __init__(self, legalLabels,inputNum, hiddenNum, outputNum, dataNum, l):
        """
        input: the number of input neurons (in this case features)
        hidden: the number of hidden neurons (should be tuned)
        output: the number of output neurons (the classifications of image)
        l: lambda
        """
        self.input = inputNum  # without bias node
        self.hidden = hiddenNum  # without bias node
        self.output = outputNum
        self.dataNum = dataNum
        self.legalLabels = legalLabels
        self.l = l

        "allocate memory for activation matrix of 1s"
        self.inputActivation = np.ones((self.input + 1, dataNum))  # add bias node
        self.hiddenActivation = np.ones((self.hidden + 1, dataNum))  # add bias node
        self.outputActivation = np.ones((self.output, dataNum))

        "allocate memory for bias vector"
        self.bias = np.ones((1, dataNum))

        "allocate memory for change matrix of 0s"
        self.inputChange = np.zeros((self.hidden, self.input + 1))
        self.outputChange = np.zeros((self.output, self.hidden + 1))

        "calculate epsilon for randomization"
        self.hiddenEpsilon = np.sqrt(6.0 / (self.input + self.hidden))
        self.outputEpsilon = np.sqrt(6.0 / (self.input + self.output))

        "allocate memory for randomized weights"
        self.inputWeights = np.random.rand(self.hidden, self.input + 1) * 2 * self.hiddenEpsilon - self.hiddenEpsilon
        self.outputWeights = np.random.rand(self.output, self.hidden + 1) * 2 * self.outputEpsilon - self.outputEpsilon

    def setLambda(self, l):
        "update lambda"
        self.l = l

    def feedForward(self, thetaVec):
        "reshape thetaVec into two weights matrices"
        self.inputWeights = thetaVec[0:self.hidden * (self.input + 1)].reshape((self.hidden, self.input + 1))
        self.outputWeights = thetaVec[-self.output * (self.hidden + 1):].reshape((self.output, self.hidden + 1))

        "hidden activation"
        hiddenZ = self.inputWeights.dot(self.inputActivation)
        self.hiddenActivation[:-1, :] = sigmoid(hiddenZ)

        "output activation"
        outputZ = self.outputWeights.dot(self.hiddenActivation)
        self.outputActivation = sigmoid(outputZ)

        "calculate J"
        costMatrix = self.outputTruth * np.log(self.outputActivation) + (1 - self.outputTruth) * np.log(
            1 - self.outputActivation)
        regulations = (np.sum(self.outputWeights[:, :-1] ** 2) + np.sum(self.inputWeights[:, :-1] ** 2)) * self.l / 2
        return (-costMatrix.sum() + regulations) / self.dataNum

    def backPropagate(self, thetaVec):
        "reshape thetaVec into two weights matrices"
        self.inputWeights = thetaVec[0:self.hidden * (self.input + 1)].reshape((self.hidden, self.input + 1))
        self.outputWeights = thetaVec[-self.output * (self.hidden + 1):].reshape((self.output, self.hidden + 1))

        "calculate lower case delta"
        outputError = self.outputActivation - self.outputTruth
        hiddenError = self.outputWeights[:, :-1].T.dot(outputError) * dsigmoid(self.hiddenActivation[:-1:])

        "calculate upper case delta"
        self.outputChange = outputError.dot(self.hiddenActivation.T) / self.dataNum
        self.inputChange = hiddenError.dot(self.inputActivation.T) / self.dataNum

        "add regulations"
        self.outputChange[:, :-1].__add__(self.l * self.outputWeights[:, :-1])
        self.inputChange[:, :-1].__add__(self.l * self.inputWeights[:, :-1])

        return np.append(self.inputChange.ravel(), self.outputChange.ravel())

    def train(self, trainData, trainLabels, validData, validLabels):
        "initialization"
        self.trainData = trainData
        self.trainLabels = trainLabels
        self.validaData = validData
        self.validLabels = validLabels

        "calculate the amount of train data"
        self.size_train = len(list(trainData))
        features_train = [];
        for datum in trainData:
            feature = list(datum.values())
            features_train.append(feature)
        train_set = np.array(features_train, np.int32)

        iteration = 100
        "after we got train_set, we calculate the activation for every datum"
        self.inputActivation[:-1, :] = train_set.transpose()
        "then we give the output truth labels for every datum"
        self.outputTruth = self.genTruthMatrix(trainLabels)
        "propagate"

        thetaVec = np.append(self.inputWeights.ravel(), self.outputWeights.ravel())
        thetaVec = opt.fmin_cg(self.feedForward, thetaVec, fprime=self.backPropagate, maxiter=iteration)
        self.inputWeights = thetaVec[0:self.hidden * (self.input + 1)].reshape((self.hidden, self.input + 1))
        self.outputWeights = thetaVec[-self.output * (self.hidden + 1):].reshape((self.output, self.hidden + 1))


    def classify(self, testData):
        "input activation"
        "for classify in case the difference of size between trainData and testData "
        self.size_test = len(list(testData))
        features_test = [];
        for datum in testData:
            feature = list(datum.values())
            features_test.append(feature)
        test_set = np.array(features_test, np.int32)
        feature_test_set = test_set.transpose()

        if feature_test_set.shape[1] != self.inputActivation.shape[1]:
            self.inputActivation = np.ones((self.input + 1, feature_test_set.shape[1]))
            self.hiddenActivation = np.ones((self.hidden + 1, feature_test_set.shape[1]))
            self.outputActivation = np.ones((self.output + 1, feature_test_set.shape[1]))
        self.inputActivation[:-1, :] = feature_test_set

        "hidden activation"
        hiddenZ = self.inputWeights.dot(self.inputActivation)
        self.hiddenActivation[:-1, :] = sigmoid(hiddenZ)

        "output activation"
        outputZ = self.outputWeights.dot(self.hiddenActivation)
        self.outputActivation = sigmoid(outputZ)
        if self.output > 1:
            return np.argmax(self.outputActivation, axis=0).tolist()
        else:
            return (self.outputActivation>0.5).ravel()

    "Checking condition matrix"
    def genTruthMatrix(self, trainLabels):
        truth = np.zeros((self.output, self.dataNum))
        for i in range(self.dataNum):
            label = trainLabels[i]
            if self.output == 1:
                truth[:,i] = label
            else:
                truth[label, i] = 1
        return truth
