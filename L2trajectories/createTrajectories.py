import numpy as np

from L2trajectories import findTrajectoriesInWindow
from L2trajectories import solveInGroups
from L2trajectories import trackletsToTrajectories
from L2trajectories import trajectoriesVis
from L2trajectories import recomputeTrajectories

# CREATETRAJECTORIES partitions a set of tracklets into trajectories.
# The third stage uses appearance grouping to reduce problem complexity;
# the fourth stage solves the graph partitioning problem for each appearance group.

def createTrajectories(traje_ops, inputTrajectories, startTime, endTime,engine):

    ## find current, old, and future tracklets
    currentTrajectoriesInd = findTrajectoriesInWindow.findTrajectoriesInWindow(inputTrajectories, startTime, endTime)
    if(currentTrajectoriesInd == None or len(currentTrajectoriesInd)==1):
        return inputTrajectories
    
    currentTrajectories = []
    for ind in currentTrajectoriesInd:
        currentTrajectories.append(inputTrajectories[ind])
        
# select tracklets that will be selected in association. For previously
# computed trajectories we select only the last three tracklets.
    inAssociation = []
    tracklets = []
    trackletLabels = []
    for i in range(len(currentTrajectories)):
        for k in range(len(currentTrajectories[i]['tracklets'])):
            tracklets.append(currentTrajectories[i]['tracklets'][k])
            trackletLabels.append(i)
            
            inAssociation.append(True)
#            if( k>= (len(currentTrajectories[i]['tracklets'])-5)):
#                inAssociation[-1] = True
                
#    solvetracklets = []
#    solvetrackletLabels = []
#    solveindex = []
#    for ind in range(len(inAssociation)):
#        if(inAssociation[ind]):
#            solveindex.append(ind)
#            solvetracklets.append(tracklets[ind])
#            solvetrackletLabels.append(trackletLabels[ind])

    ## solve the graph partitioning problem for each appearance group
#    result = solveInGroups.solveInGroups(traje_ops, solvetracklets, solvetrackletLabels,engine)

    result = solveInGroups.solveInGroups(traje_ops, tracklets, trackletLabels,engine)
    
    ## merge back solution. Tracklets that were associated are now merged back
    ## with the rest of the tracklets that were sharing the same trajectory

#    labels = list(trackletLabels)
#    for ind,val in enumerate(solveindex):
#        labels[val] = result['labels'][ind]
#        
#    count = 0
#    trackletLabels = np.array(trackletLabels)
#    labels = np.array(labels)
#    for ind in range(len(inAssociation)):
#        if(inAssociation[ind]):
#            indices = np.where(trackletLabels == trackletLabels[ind]) 
#            labels[indices] = result['labels'][count]
#            count = count + 1
            
    ## merge co-identified tracklets to extended tracklets
#    newTrajectories = trackletsToTrajectories.trackletsToTrajectories(tracklets, labels)

    newTrajectories = trackletsToTrajectories.trackletsToTrajectories(tracklets, result['labels'])

#    smoothTrajectories = recomputeTrajectories.recomputeTrajectories(newTrajectories)

    outputTrajectories = inputTrajectories
    for indet in sorted(currentTrajectoriesInd,reverse=True):
        del outputTrajectories[indet]
    outputTrajectories.extend(newTrajectories)

#    trajectoriesVis.trajectoriesVis(outputTrajectories)
    return outputTrajectories


