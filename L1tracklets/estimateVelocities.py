### This function estimates the velocity of a detection by calculating the component-wise
### median of velocities required to reach a specified number of nearest neighbors.
### Neighbors that exceed a specified speed limit are not considered.

import sys
import os
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

def estimateVelocities(detections,detectionCenters,searchRangeFrames,frame_start,frame_end,nearest_neighbors,speedLimit):
    estimatedVelocities =[]

    ###Compute all pairwise distances 
    pairDistance = pairwise_distances(detectionCenters,metric='euclidean')
    numDetections = len(detections)
    
    ### Estimate the velocity of each detection
    for currentDtectionIndex in range(0,numDetections):
        currentFrame = searchRangeFrames[currentDtectionIndex]
        velocities = []
        #  For each time instant in a small time neighborhood find the nearest detection in space
        for frame in range(currentFrame-nearest_neighbors,currentFrame+nearest_neighbors):
            # Skip original frame
            if(currentFrame == frame):
                continue
                
            # Skip if no detections in the current frame
            if frame not in searchRangeFrames:
                continue
            
            # Find detection closest to the current detection
            detectionsAtThisTimeInstant = np.where(searchRangeFrames == frame)   
            distancesAtThisTimeInstant = pairDistance[currentDtectionIndex][detectionsAtThisTimeInstant]
            minIndex = np.argmin(distancesAtThisTimeInstant)
            targetDetectionIndex = detectionsAtThisTimeInstant[0][minIndex]
           # print('detectionsAtThisTimeInstant',targetDetectionIndex)
            estimatedVelocity = detectionCenters[targetDetectionIndex] - detectionCenters[currentDtectionIndex]
            estimatedVelocity = estimatedVelocity/(searchRangeFrames[targetDetectionIndex]-searchRangeFrames[currentDtectionIndex])

            # Check if speed limit is violated
            estimatedSpeed = np.sqrt(np.sum(np.square(estimatedVelocity)))
            if(estimatedSpeed>speedLimit):
                continue
            velocities.append(estimatedVelocity)

        if(len(velocities) == 0):
            velocities.append([0,0])
        velocities = np.array(velocities)
        ### Estimate the velocity
        estimatedVelocities.append([np.mean(velocities[:,0]),np.mean(velocities[:,1])])
    return estimatedVelocities , pairDistance

