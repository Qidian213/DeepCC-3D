# This function computes the motion affinities given a set of detections.
# A simple motion prediction is performed from a source detection to
# a target detection to compute the prediction error.

import sys
import os
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

def motionAffinity_3d(spatialGroupDetection_3d,spatialGroupDetectionFrames,spatialGroupEstimatedVelocity_3d,speed_limit,beta):

    numDetections  = len(spatialGroupDetection_3d)
    impossibilityMatrix = np.zeros((len(spatialGroupDetectionFrames),len(spatialGroupDetectionFrames)))
    spatialGroupDetectionFrames = np.transpose([spatialGroupDetectionFrames])
    frameDifference = pairwise_distances(spatialGroupDetectionFrames,metric='euclidean')
    
    tmp_velocityX = np.transpose([spatialGroupEstimatedVelocity_3d[:,0]])
    tmp_velocityY = np.transpose([spatialGroupEstimatedVelocity_3d[:,1]])
    tmp_velocityZ = np.transpose([spatialGroupEstimatedVelocity_3d[:,2]])
    tmp_centerX = np.transpose([spatialGroupDetection_3d[:,0]])
    tmp_centerY = np.transpose([spatialGroupDetection_3d[:,1]])
    tmp_centerZ = np.transpose([spatialGroupDetection_3d[:,2]])
    
    velocityX = np.tile(tmp_velocityX,(1, numDetections))
    velocityY = np.tile(tmp_velocityY,(1, numDetections))
    velocityZ = np.tile(tmp_velocityZ,(1, numDetections))
    centerX   = np.tile(tmp_centerX,(1, numDetections))
    centerY   = np.tile(tmp_centerY,(1, numDetections))
    centerZ   = np.tile(tmp_centerZ,(1, numDetections))

    errorXForward = centerX + np.multiply(velocityX,frameDifference) - np.transpose(centerX)
    errorYForward = centerY + np.multiply(velocityY,frameDifference) - np.transpose(centerY)
    errorZForward = centerZ + np.multiply(velocityZ,frameDifference) - np.transpose(centerZ)
    
    errorXBackward = np.transpose(centerX) + np.multiply(np.transpose(velocityX),-np.transpose(frameDifference)) - centerX
    errorYBackward = np.transpose(centerY) + np.multiply(np.transpose(velocityY),-np.transpose(frameDifference)) - centerY
    errorZBackward = np.transpose(centerZ) + np.multiply(np.transpose(velocityZ),-np.transpose(frameDifference)) - centerZ
    
    errorForward = np.sqrt(np.power(errorXForward,2)+np.power(errorYForward,2)+np.power(errorZForward,2))
    errorBackward = np.sqrt(np.power(errorXBackward,2)+np.power(errorYBackward,2)+np.power(errorZBackward,2))
    
    ### Only upper triangular part is valid
    predictionError = np.minimum(errorForward, errorBackward)
    predictionError = np.triu(predictionError) + np.transpose(np.triu(predictionError))

    ### Check if speed limit is violated 
    xDiff = centerX - np.transpose(centerX)
    yDiff = centerY - np.transpose(centerY)
    zDiff = centerZ - np.transpose(centerZ)
    distanceMatrix = np.sqrt(np.power(xDiff,2)+np.power(yDiff,2)+np.power(zDiff,2))
    
    maxRequiredSpeedMatrix = np.divide(distanceMatrix,np.abs(frameDifference))
    predictionError[maxRequiredSpeedMatrix > speed_limit] = float('inf')
    impossibilityMatrix[maxRequiredSpeedMatrix > speed_limit] = 1;
    motionScores = 1 - beta*predictionError;
    
    return motionScores, impossibilityMatrix,frameDifference
