import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from L2trajectories import getTrackletFeatures

def frameDiffTest(ind,centerFrame):
    frame1 = centerFrame[ind]
    diiftmp = []
    for ceframe in centerFrame:
        diiftmp.extend(ceframe[0]-frame1)
    return diiftmp

def overlapTest(ind,intervals):
    interval1 = intervals[ind]
    interval1 = np.array(interval1)
    intervals = np.array(intervals)
    
    duration1 = interval1[1] - interval1[0]
    duration2 = [ intter[1]-intter[0] for intter in intervals]
    
    i1 = np.tile(interval1,(len(intervals),1))
    tmpminmax = np.hstack((i1,intervals))
    unionMin = np.min(tmpminmax,axis=1)
    unionMax = np.max(tmpminmax,axis=1)
    
    overlap = [1.0 if(duration1+duration2_val-unMax+unMin)>=0 else 0.0 for (duration2_val,unMax,unMin) in zip(duration2,unionMax,unionMin)]
    return overlap

def getSpaceTimeAffinity(tracklets, beta1, speedLimit, indifferenceLimit):

    numTracklets = len(tracklets)

#    centersWorld = [] ### [[X,Y,T],[X,Y,T]....]
#    centersView = [] ### [[X,Y,T],[X,Y,T]....]
#    startpoint = []  ### [[x,y],[x,y]....]
#    endpoint = []    ### [[x,y],[x,y]....]
#    intervals = []   ### [[frame_0,frame_n],[frame_0,frame_n]....]
#    duration = []   ### [frame_n-frame_0,frame_n-frame_0,....]
#    velocity = []   ### [[vx,vy],[vx,vy],....]

    centersWorld, centersView, startpoint, endpoint, intervals, duration, velocity = getTrackletFeatures.getTrackletFeatures(tracklets)
    
    centerFrame = [[round((interval[0]+interval[1])/2)] for interval in intervals]
    centerFrame = np.array(centerFrame)

    frameDifference = []
    for i in range(len(centerFrame)):
        diffframe = frameDiffTest(i,centerFrame)
        frameDifference.append(diffframe)
    frameDifference = np.array(frameDifference)

    overlapping = []
    for i in range(len(intervals)):
        overtest = overlapTest(i,intervals)
        overlapping.append(overtest)
    overlapping = np.array(overlapping)

    startpoint = np.array(startpoint)
    endpoint = np.array(endpoint)
    centers = 0.5*(startpoint+endpoint)
    centersDistance = pairwise_distances(centers,metric='euclidean')

    frameDifferenceV = frameDifference.copy()
    frameDifferenceV[frameDifferenceV <0] = 0
    frameDifferenceV[frameDifferenceV >0] =1
    v = frameDifferenceV + overlapping
    v[v>0] = 1

    centersDistance_mer = centersDistance.copy()
    centersDistance_mer[centersDistance_mer < 5] = 1
    centersDistance_mer[centersDistance_mer >= 5] = 0
    merging = np.multiply(centersDistance_mer,overlapping)

    velocity = np.array(velocity)
    tmp_velocityX = np.transpose([velocity[:,0]])
    tmp_velocityY = np.transpose([velocity[:,1]])
    
    velocityX = np.tile(tmp_velocityX,(1, numTracklets))
    velocityY = np.tile(tmp_velocityY,(1, numTracklets))

    tmp_centerX = np.transpose([centers[:,0]])
    tmp_centerY = np.transpose([centers[:,1]])
    
    startX = np.tile(tmp_centerX,(1, numTracklets))
    startY = np.tile(tmp_centerY,(1, numTracklets))
    endX = np.tile(tmp_centerX,(1, numTracklets))
    endY = np.tile(tmp_centerY,(1, numTracklets))

    errorXForward = endX + np.multiply(velocityX,frameDifference) - np.transpose(startX)
    errorYForward = endY + np.multiply(velocityY,frameDifference) - np.transpose(startY)
    
    errorXBackward = np.transpose(startX) + np.multiply(np.transpose(velocityX),-np.transpose(frameDifference)) - endX
    errorYBackward = np.transpose(startY) + np.multiply(np.transpose(velocityY),-np.transpose(frameDifference)) - endY
    
    errorForward = np.sqrt(np.power(errorXForward,2)+np.power(errorYForward,2))
    errorBackward = np.sqrt(np.power(errorXBackward,2)+np.power(errorYBackward,2))

    ## check if speed limit is violated
    xDiff = endX - np.transpose(startX)
    yDiff = endY - np.transpose(startY)
    distanceMatrix = np.sqrt(np.power(xDiff,2)+np.power(yDiff,2))
    maxSpeedMatrix = np.divide(distanceMatrix,np.abs(frameDifference))

    violators = np.zeros_like(maxSpeedMatrix)
    violators[maxSpeedMatrix > speedLimit] = 1
    violators = np.multiply(violators,v)
    violators = violators + np.transpose(violators)
    
    ## build impossibility matrix
    impossibilityMatrix = np.zeros((numTracklets,numTracklets))
    impossibilityMatrix[violators == 1] = 1
    impossibilityMatrix[merging == 1] = 0
    impossibilityMatrix[overlapping == 1] = 1
    impossibilityMatrix[merging == 1] = 0

    ## this is a symmetric matrix, although tracklets are oriented in time
    errorMatrix = np.minimum(errorForward, errorBackward)
    errorMatrix = np.multiply(errorMatrix,v)
    errorMatrix = np.multiply(errorMatrix,v)
    errorMatrix = errorMatrix + np.transpose(errorMatrix)
    errorMatrix[violators==1] = float('inf')

    ## compute indifference matrix
    frameDifferenceTmp = frameDifference.copy()
    frameDifferenceTmp[frameDifferenceTmp<=0] = 0
    frameDifferenceTmp[frameDifferenceTmp>0] = 1
    timeDifference = np.multiply(frameDifference,frameDifferenceTmp)
    timeDifference = timeDifference + np.transpose(timeDifference)
    params = [0.1,indifferenceLimit/2]
    indiffMatrix = 1 - 1./(1+np.exp(-params[0]*(timeDifference - params[1])))
    
    ## compute space-time affinities
    stAffinity = 1 - beta1*errorMatrix
    stAffinity[stAffinity<0] = 0

    return stAffinity, impossibilityMatrix, indiffMatrix 

