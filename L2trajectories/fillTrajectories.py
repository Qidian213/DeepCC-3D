import numpy as np

# This function adds points by interpolation to the resulting trajectories,
# so that trajectories are complete and results are less influenced by
# false negative detections

## detections [frame, traj_id,x,y,w,h]
def fillTrajectories(detections):
    detectionsUpdated = []
    
    detections = np.array(detections)
    detections = detections[detections[:,5].argsort(),]
    detections = detections[detections[:,4].argsort(),]
    detections = detections[detections[:,3].argsort(),]
    detections = detections[detections[:,2].argsort(),]
    detections = detections[detections[:,0].argsort(),]
    
    personIDs = np.unique(detections[:,1])
    
    count = 0
    
    for i in range(len(personIDs)):
        personID = personIDs[i]
        
        person_det = np.where(detections[:,1]==personID)
        relevantDetections = detections[person_det]

        startFrame = np.min(relevantDetections[:,0])
        endFrame = np.max(relevantDetections[:,0])

        frame_interval = np.linspace(startFrame,endFrame,endFrame-startFrame+1)
        missingFrames = np.setdiff1d(frame_interval, relevantDetections[:,0])
        
        if(len(missingFrames)==0):
            continue
            
        frameDiff = np.diff(missingFrames)
        frameDiff_ind = np.where(frameDiff>1)
        
        startInd = [0]
        startInd.extend((np.array(frameDiff_ind)+1).tolist()[0])
        
        endInd = []
        endInd.extend(np.array(frameDiff_ind).tolist()[0])
        endInd.append(len(frameDiff))
        
        startInd = np.array(startInd)
        endInd = np.array(endInd)

        for k in range(len(startInd)):
            interpolatedDetections = np.zeros((int(missingFrames[endInd[k]] - missingFrames[startInd[k]] + 1) , len(detections[0])))
            interpolatedDetections[:,1] = personID
            
            for j in range(int(missingFrames[endInd[k]] - missingFrames[startInd[k]] + 1)):
                interpolatedDetections[j][0] = missingFrames[startInd[k]]+j
            
            predet_index = np.where(relevantDetections[:,0] == (missingFrames[startInd[k]]-1))
            postdet_index = np.where(relevantDetections[:,0] == (missingFrames[endInd[k]] +1))
            
            preDetection = relevantDetections[predet_index][0]
            postDetection = relevantDetections[postdet_index][0]
            
            for c in range(2,len(detections[0])):
                insertval = np.linspace(preDetection[c],postDetection[c],int(missingFrames[endInd[k]] - missingFrames[startInd[k]] + 1))
                for m in range(int(missingFrames[endInd[k]] - missingFrames[startInd[k]] + 1)):
                    interpolatedDetections[m][c] = insertval[m]
                    
            detectionsUpdated.extend(interpolatedDetections.tolist())

    return detectionsUpdated





