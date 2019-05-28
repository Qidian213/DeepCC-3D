import sys
import os
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

def getAppearanceSubMatrix(elements,features,threshold):
    feature_id = features[elements]
    ###Compute all pairwise distances 
    pairDistance = pairwise_distances(feature_id,metric='euclidean')
    correlation = (threshold - pairDistance)/threshold
    return correlation
