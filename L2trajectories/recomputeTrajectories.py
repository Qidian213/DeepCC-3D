import numpy as np

#   RECOMPUTETRAJECTORIES Summary of this function goes here
#   Detailed explanation goes here
def recomputeTrajectories(trajectories):
    segmentLength = 20
    
    for i in range(len(trajectories)):
        segmentStart = trajectories[i]['segmentStart']
        segmentEnd = trajectories[i]['segmentEnd']
        
        numSegments = (segmentEnd + 1 - segmentStart) / segmentLength

        alldata = []
        for k in range(len(trajectories[i]['tracklets'])):
            alldata.extend(trajectories[i]['tracklets'][k]['data'])
        alldata = np.array(alldata)
        alldata = alldata[alldata[:,0].argsort(),]
        alldata = alldata[alldata[:,1].argsort(),]
        
        tmpdata, uniqueRows = np.unique(alldata[:,0], return_index=True)
        uniqueRows = uniqueRows.astype(np.int32)
        
        alldata = alldata[uniqueRows]
        dataFrames = alldata[:,0]
        dataFrames = np.array(dataFrames)
        
        frames = np.linspace(segmentStart,segmentEnd,segmentEnd-segmentStart+1)
        
        tmp_frames = []
        tmp_frames.append(np.min(dataFrames))
        
        segnum = (frames[-1] - frames[0] - segmentLength/2)/segmentLength
        segint = np.linspace(frames[0]+ segmentLength/2,frames[-1],int(segnum))
        segint = list(segint)
        tmp_frames.extend(segint)
        tmp_frames.append(np.max(dataFrames))
        print(segmentStart)
        print(segmentEnd)
        print(frames[0])
        print(frames[-1])
        print(tmp_frames)
#        interestingFrames = round([min(dataFrames), frames(1) + segmentLength/2:segmentLength:frames(end),  max(dataFrames)]);
        

