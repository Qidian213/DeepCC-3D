### CREATETRACKLETS This function creates short tracks composed of several detections.
###   In the first stage our method groups detections into space-time groups.
###   In the second stage a Binary Integer Program is solved for every space-time group.

import sys
import os
import numpy as np
from sklearn.cluster import SpectralClustering ,AffinityPropagation

from external import utils
from L1tracklets import estimateVelocities
from L1tracklets import getSpatialGroupIDs
from L1tracklets import getAppearanceSubMatrix
from L1tracklets import motionAffinity
from L1tracklets import smoothTracklets
from L1tracklets import trackletsVis
from L1tracklets import trackletsVis2

def create_tracklets(track_ops,feature_bb,frame_start,frame_end,engine):  
    tracklet = []

    detections = [x[0:6] for x in feature_bb]
    detections_3d = [x[6:9] for x in feature_bb]
    features = [x[9:] for x in feature_bb]
    features = np.array(features)
    
    ###Find detections in search range
    searchRangeFrames = [int(x[1]) for x in detections]
    searchRangeFrames = np.array(searchRangeFrames)
    
#### Divide detections in space-groups
    totalLabels     = 0
    currentInterval = 0
    
    # Compute bounding box centeres
    detectionCenters = utils.getBoundingBoxCenters(detections)
    detectionCenters = np.array(detectionCenters)
    
    # Estimate velocities
    estimatedVelocity , pairwiseDistances = estimateVelocities.estimateVelocities(detections,detectionCenters,searchRangeFrames,frame_start,frame_end,track_ops['nearest_neighbors'],track_ops['speed_limit'])
    estimatedVelocity = np.array(estimatedVelocity)
    
    # Spatial groupping
    spatialGroupIDs = getSpatialGroupIDs.getSpatialGroupIDs(True,searchRangeFrames,pairwiseDistances,track_ops)

#### SOLVE A GRAPH PARTITIONING PROBLEM FOR EACH SPATIAL GROUP
    print("Creating tracklets: solving space-time groups")
    
    for spatialGroupID in range(1,max(spatialGroupIDs)+1):
        print('spatialGroupID: ',spatialGroupID)
        elements = np.where(spatialGroupIDs == spatialGroupID) 

        # Create an appearance affinity matrix and a motion affinity matrix
        appearanceCorrelation = getAppearanceSubMatrix.getAppearanceSubMatrix(elements,features,track_ops['threshold'])
        spatialGroupDetectionCenters = detectionCenters[elements]
        spatialGroupDetectionFrames = searchRangeFrames[elements]
        spatialGroupEstimatedVelocity = estimatedVelocity[elements]
        
        motionCorrelation, impMatrix,intervalDistance = motionAffinity.motionAffinity(spatialGroupDetectionCenters,spatialGroupDetectionFrames,spatialGroupEstimatedVelocity,track_ops['speed_limit'], track_ops['beta'])
        
        # Combine affinities into correlations
        discountMatrix = np.minimum(1, -np.log(intervalDistance/track_ops['window_width']))
        
        correlationMatrix = np.multiply(motionCorrelation,discountMatrix)+appearanceCorrelation
        correlationMatrix[impMatrix==1] =  -float('inf')                                       ####check point
        
        # Solve the graph partitioning problem
        
        #print('spatialGroupID: ',spatialGroupID)
        correlationMatrix = correlationMatrix.tolist()
        labels = engine.KernighanLin(correlationMatrix,len(correlationMatrix))
        labels =  np.array(labels,dtype = float)
        labels = labels+totalLabels
        totalLabels = np.max(labels)
        identities = labels
        if len(elements[0])>1:
            for k,v in enumerate(elements[0]):
                feature_bb[v][0] = identities[k][0]
        else:
            feature_bb[int(elements[0][0])][0] = identities
            
#        trackletsVis.trackletsVis(spatialGroupDetectionCenters,labels)
    trackletsVis2.trackletsVis2(feature_bb)
    
    ### FINALIZE TRACKLETS
    ### Fit a low degree polynomial to include missing detections and smooth the tracklet
    smoothedTracklets = smoothTracklets.smoothTracklets(feature_bb,totalLabels,frame_start,track_ops['window_width'], track_ops['min_length'], currentInterval)
    
    for i in range(len(smoothedTracklets)):
        smoothedTracklets[i]['id'] = i+1
        smoothedTracklets[i]['ids'] = i+1
    
    return smoothedTracklets





