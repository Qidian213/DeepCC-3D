import numpy as np

### startTime, endTime sliding window

def findTrajectoriesInWindow(inputTrajectories, startTime, endTime):
    trajectoriesInWindow = []
    if(inputTrajectories == None):
        return 

    for i in range(len(inputTrajectories)):
        if(inputTrajectories[i]['endFrame'] >= startTime and inputTrajectories[i]['startFrame']<=endTime):
            trajectoriesInWindow.append(i)
    return trajectoriesInWindow
#    print('trajectoriesInWindow',trajectoriesInWindow)
