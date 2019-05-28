import numpy as np
from external import utils

def smoothTracklets(feature_bb,numTracklets,segmentStart,segmentInterval, minTrackletLength, currentInterval):
    feature_bb = np.array(feature_bb)
    
    smoothedTracklets = []
    for i in range(1,int(numTracklets)+1):
        mask = []
        for j in range(len(feature_bb)):
            if feature_bb[j][0] == i:
                mask.append(j)
        mask = np.array(mask)
        detections_fetures = feature_bb[mask]

        ### Reject tracklets of short length
        start = detections_fetures[0][1]
        finish = detections_fetures[-1][1]
        
        if(len(detections_fetures)<minTrackletLength or (finish-start)<minTrackletLength):
            continue
            
        intervalLength = finish - start +1
        datapoints = np.linspace(start,finish,intervalLength)
     #   print(start,finish,intervalLength)
        frames = [int(x[1]) for x in detections_fetures]
        frames = np.array(frames)
        currentTracklet = np.zeros((int(intervalLength),6))
        for val in range(int(intervalLength)):
            currentTracklet[val][1] = i
            currentTracklet[val][0] = start+val
        
        ### Fit left, top, right, bottom, xworld, yworld
        for n in range(2,6):
            points = [defe[n] for defe in detections_fetures]
            points = np.array(points)
            p = np.polyfit(frames,points,1)
            newpoints = np.polyval(p,datapoints)
            for val in range(int(intervalLength)):
                currentTracklet[val][n] = newpoints[val]

        ###  Compute appearance features
        medianFeature_det =  np.mean(detections_fetures, axis=0)  #### or median
        medianFeature = medianFeature_det[6:]
        centers = medianFeature_det[:6]
        centerPoint = [centers[2]+0.5*centers[4],centers[3]+0.5*centers[5]]
        centerPoint = np.array(centerPoint)
        centerPointWorld = 1
        
        ### Add to tracklet list
        tmp_smoothTracklets = {}
        tmp_smoothTracklets['feature'] = medianFeature
        tmp_smoothTracklets['center'] = centerPoint
        tmp_smoothTracklets['centerWorld'] = centerPointWorld
        tmp_smoothTracklets['data'] = currentTracklet
        tmp_smoothTracklets['realdata_features'] = detections_fetures
        tmp_smoothTracklets['mask'] = mask
        tmp_smoothTracklets['startFrame'] = start
        tmp_smoothTracklets['endFrame'] = finish
        tmp_smoothTracklets['interval'] = currentInterval
        tmp_smoothTracklets['segmentStart'] = segmentStart
        tmp_smoothTracklets['segmentInterval'] = segmentInterval
        tmp_smoothTracklets['segmentEnd'] = segmentStart+segmentInterval-1
        tmp_smoothTracklets['id'] = i
        tmp_smoothTracklets['ids'] = i
        smoothedTracklets.append(tmp_smoothTracklets)
    return smoothedTracklets











