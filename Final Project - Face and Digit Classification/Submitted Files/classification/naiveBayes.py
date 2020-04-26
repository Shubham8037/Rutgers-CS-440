# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import collections
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.intial = None  # The intial setting
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.count = None # Total count
    self.sec = None
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

  def check(self, out):
    prob = dict(collections.Counter(out))
    for k in prob.keys():
        prob[k] = prob[k] / float(len(out))
    return prob

    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k


  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid): 
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    intial = dict(collections.Counter(trainingLabels))  # Get the number of training labels

    for k in intial.keys():
        intial[k] = intial[k] / float(len(trainingLabels))

    sec = dict()  # Intialize a dictionary for sec

    for x, prob in intial.items():  # For every item we create a new dict
        sec[x] = collections.defaultdict(list) # Create the sec of default dictionary list

    for x, prob in intial.items():
        first = list()
        for i, ptr in enumerate(trainingLabels):        # go through the traningLabels and check the indexs and append
            if x == ptr:                                # Check the index 
                first.append(i)

        second = list()

        for i in first:     # Second is list that will contain training data based on labels
            second.append(trainingData[i])

        for y in range(len(second)):    # Now we populate the dictionary with the correct label and the data
            for k, ptr in second[y].items():
                sec[x][k].append(ptr)

    count = [a for a in intial] # Get the total count

    for x in count:     
        for k, ptr in second[x].items():
            sec[x][k] = self.check(sec[x][k])   # Get the probabilties for Naive Bayes

    self.intial = intial    # Update the intial
    self.count = count      # Update the count
    self.sec = sec          # Update the second list with the training label and training data

    "*** YOUR CODE HERE ***"

    #util.raiseNotDefined()
        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    
    "*** YOUR CODE HERE ***"   
    for x in self.count:    
        probs = self.intial[x]  # Get the probabilty 

        for k, ptr in datum.items():
            nf = self.sec[x][k]     # Get the data we need from the sec dict
            probs = probs + math.log(nf.get(datum[k], 0.01)) # Calculate the probability 

        logJoint[x] = probs  # Add the new probability back to the log Joint list 

    #util.raiseNotDefined()
    
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
       
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds
    

    
      
