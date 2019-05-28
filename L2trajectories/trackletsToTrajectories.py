import cv2
import numpy as np

def trackletsToTrajectories(Tracklets,labels):
    trajectories = []
    uniqueLabels = np.unique(labels)

    for label in uniqueLabels:
        trackletIndices = np.where(labels == label)
        trajectory = {'tracklets':[],'startFrame':float('inf'),'endFrame':float('-inf'),'segmentStart':float('inf'),'segmentEnd':float('-inf')}

        for ind in trackletIndices[0]:
            trajectory['tracklets'].append(Tracklets[int(ind)])
            trajectory['startFrame'] = trajectory['startFrame'] if trajectory['startFrame'] < Tracklets[int(ind)]['startFrame'] else Tracklets[int(ind)]['startFrame']
            trajectory['endFrame'] = trajectory['endFrame'] if trajectory['endFrame'] > Tracklets[int(ind)]['endFrame'] else Tracklets[int(ind)]['endFrame']
            trajectory['segmentStart'] = trajectory['segmentStart'] if trajectory['segmentStart'] < Tracklets[int(ind)]['segmentStart'] else Tracklets[int(ind)]['segmentStart']
            trajectory['segmentEnd'] = trajectory['segmentEnd'] if trajectory['segmentEnd'] > Tracklets[int(ind)]['segmentEnd'] else Tracklets[int(ind)]['segmentEnd']
            trajectory['feature'] = Tracklets[int(ind)]['feature']
        trajectories.append(trajectory)
        
    return trajectories
