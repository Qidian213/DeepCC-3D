# This function computes the motion affinities given a set of detections.
# A simple motion prediction is performed from a source detection to
# a target detection to compute the prediction error.

import sys
import os
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

def motionAffinity(spatialGroupDetectionCenters,spatialGroupDetectionFrames,spatialGroupEstimatedVelocity,speed_limit,beta):

    numDetections  = len(spatialGroupDetectionCenters)
    impossibilityMatrix = np.zeros((len(spatialGroupDetectionFrames),len(spatialGroupDetectionFrames)))
    spatialGroupDetectionFrames = np.transpose([spatialGroupDetectionFrames])
    frameDifference = pairwise_distances(spatialGroupDetectionFrames,metric='euclidean')
    
    tmp_velocityX = np.transpose([spatialGroupEstimatedVelocity[:,0]])
    tmp_velocityY = np.transpose([spatialGroupEstimatedVelocity[:,1]])
    tmp_centerX = np.transpose([spatialGroupDetectionCenters[:,0]])
    tmp_centerY = np.transpose([spatialGroupDetectionCenters[:,1]])
    
    velocityX = np.tile(tmp_velocityX,(1, numDetections))
    velocityY = np.tile(tmp_velocityY,(1, numDetections))
    centerX   = np.tile(tmp_centerX,(1, numDetections))
    centerY   = np.tile(tmp_centerY,(1, numDetections))
    
    errorXForward = centerX + np.multiply(velocityX,frameDifference) - np.transpose(centerX)
    errorYForward = centerY + np.multiply(velocityY,frameDifference) - np.transpose(centerY)
    
    errorXBackward = np.transpose(centerX) + np.multiply(np.transpose(velocityX),-np.transpose(frameDifference)) - centerX
    errorYBackward = np.transpose(centerY) + np.multiply(np.transpose(velocityY),-np.transpose(frameDifference)) - centerY
    
    errorForward = np.sqrt(np.power(errorXForward,2)+np.power(errorYForward,2))
    errorBackward = np.sqrt(np.power(errorXBackward,2)+np.power(errorYBackward,2))
    
    ### Only upper triangular part is valid
    predictionError = np.minimum(errorForward, errorBackward)
    predictionError = np.triu(predictionError) + np.transpose(np.triu(predictionError))

    ### Check if speed limit is violated 
    xDiff = centerX - np.transpose(centerX)
    yDiff = centerY - np.transpose(centerY)
    distanceMatrix = np.sqrt(np.power(xDiff,2)+np.power(yDiff,2))
    
    maxRequiredSpeedMatrix = np.divide(distanceMatrix,np.abs(frameDifference))
    predictionError[maxRequiredSpeedMatrix > speed_limit] = float('inf')
    impossibilityMatrix[maxRequiredSpeedMatrix > speed_limit] = 1;
    motionScores = 1 - beta*predictionError;
    
    return motionScores, impossibilityMatrix,frameDifference
