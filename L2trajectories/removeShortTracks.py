import numpy as np

#This function removes short tracks that have not been associated with any
#trajectory. Those are likely to be false positives.

def removeShortTracks(detections, cutoffLength):
    detectionsUpdated = []

    detections = np.array(detections)
    detections = detections[detections[:,1].argsort(),]
    detections = detections[detections[:,0].argsort(),]

    personIDs = np.unique(detections[:,1])

    traje_num = 0
    for i in range(len(personIDs)):
        personID = personIDs[i]
        
        person_det_index = np.where(detections[:,1]==personID)
        
        if(len(person_det_index[0])<cutoffLength):
            continue
            
        person_dets = detections[person_det_index]
        person_dets[:,1] = traje_num
        detectionsUpdated.extend(person_dets.tolist())
        traje_num = traje_num + 1
        
    return detectionsUpdated
