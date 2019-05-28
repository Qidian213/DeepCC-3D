import numpy as np
from collections import Counter
from scipy.cluster.vq import vq,kmeans
from sklearn.metrics.pairwise import pairwise_distances
from L2trajectories import getAppearanceMatrix
from L2trajectories import getSpaceTimeAffinity
from L2trajectories import mergeResults

def solveInGroups(traje_ops, tracklets, labels,engine):
    if(len(tracklets)< traje_ops['appearance_groups']):
        traje_ops['appearance_groups'] =1
        
    result = {}
    result['labels']       = []
    result['observations'] = []
    
    if(len(tracklets)<=0):
        return result
    
    featureVectors = []
    for tracklet in tracklets:
        featureVectors.append(tracklet['feature'])
        
    appearanceGroups = np.ones(len(featureVectors),int)
    featureVectors = np.array(featureVectors)
    
    ### adaptive number of appearance groups
    if(traje_ops['appearance_groups']==0):
       ##### Increase number of groups until no group is too large to solve 
        while True:
            traje_ops['appearance_groups'] = traje_ops['appearance_groups'] +1
            centroid = kmeans(featureVectors,traje_ops['appearance_groups'],iter=10)[0]
            appearanceGroups = vq(featureVectors,centroid)[0] 
            count_dict = Counter(appearanceGroups)
            max_key = max(count_dict,key = count_dict.get) 
            if count_dict[max_key] <=150:
                break
    else:
        centroid = kmeans(featureVectors,2,iter=10)[0]
        appearanceGroups = vq(featureVectors,centroid)[0] 

    allGroups = np.unique(appearanceGroups)
    
    smoothedTracklets = []
    labels2d = []
    for val in labels:
        labels2d.append([val])
    labels2d = np.array(labels2d)

    result_appearance = []
    for ind in allGroups:
        print('merging tracklets in appearance group:',ind)
        group = ind
        indices = np.where(appearanceGroups == group) 
#        print('indices ',indices)
        labelsDistance = pairwise_distances(labels2d[indices],labels2d[indices],metric='euclidean')
        
        ## compute appearance and spacetime scores
        appearanceAffinity = getAppearanceMatrix.getAppearanceMatrix(featureVectors[indices],traje_ops['threshold'])
        
        tracklets_affinity = []
        for inaff in indices[0]:
            tracklets_affinity.append(tracklets[inaff])
            
        spacetimeAffinity, impossibilityMatrix, indifferenceMatrix = getSpaceTimeAffinity.getSpaceTimeAffinity(tracklets_affinity,traje_ops['beta'],traje_ops['speed_limit'],traje_ops['indifference_time'])

        ## compute the correlation matrix
        correlationMatrix = appearanceAffinity + spacetimeAffinity - 1
        correlationMatrix = np.multiply(correlationMatrix , indifferenceMatrix)
   
        correlationMatrix[impossibilityMatrix == 1] = float('-inf')
        for i in range(len(labelsDistance)):
            for j in range(len(labelsDistance[0])):
                if(labelsDistance[i][j]==0):
                    correlationMatrix[i][j] = 1.0
        
        ## solve the optimization problem
        correlationMatrix = correlationMatrix.tolist()
        labels = engine.KernighanLin(correlationMatrix,len(correlationMatrix))
        labels = np.array(labels,dtype = float)
        labels = np.ravel(labels)

        tmp_result = {}
        tmp_result['labels'] = labels
        tmp_result['observations'] = list(indices[0])
        result_appearance.append(tmp_result)
        
    ## collect independent solutions from each appearance group
#    result = {}
#    result['labels']       = []
#    result['observations'] = []
    for i in range(len(allGroups)):
        result = mergeResults.mergeResults(result,result_appearance[i])
    
    relabels = result['labels']
    reobservations = result['observations']
    relabels = np.array(relabels)
    reobservations = np.array(reobservations)
    
    inds = np.argsort(reobservations)
    reobservations = reobservations[inds]
    relabels = relabels[inds]
    
    reresult = {}
    reresult['labels'] = relabels
    reresult['observations'] = reobservations

    return reresult

