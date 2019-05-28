import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

def getAppearanceMatrix(featureVectors,threshold):
    ### Computes the appearance affinity matrix
    pairDistance = pairwise_distances(featureVectors,metric='euclidean')
    correlation = (threshold - pairDistance)/threshold
    return correlation
