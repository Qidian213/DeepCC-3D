import numpy as np

#TRAJECTORIESTOTOP Summary of this function goes here
#   Detailed explanation goes here
def trajectoriesToTop(trajectories):
    data = []
    
    for i in range(len(trajectories)):
        traj = trajectories[i]
        for k in range(len(traj['tracklets'])):
            newdata = traj['tracklets'][k]['data']
            newdata[:,1] = i
            data.extend(newdata)
            
    return data
    
