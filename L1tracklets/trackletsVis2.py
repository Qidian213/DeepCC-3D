import cv2
import numpy as np
import matplotlib.pyplot as plt  
import random

global tra_num
tra_num = 0

def trackletsVis2(feature_bb):
    image = cv2.imread("data/vis.jpg")
    
    feature_bb = np.array(feature_bb)
    trackIDs = np.unique(feature_bb[:,0])
    
    for i in range(len(trackIDs)):
        trackID = trackIDs[i]
        color = [random.randint(0,255) for cl in range(3)]
        
        track_det = np.where(feature_bb[:,0]==trackID)
        relevantDetections = feature_bb[track_det]
        
        for det in relevantDetections:
            if(det[2]==0 and det[3]==0):
                continue
            cv2.circle(image, (int(det[2]+0.5*det[4]),int(det[3]+0.5*det[5])), 1, color, 4)

    global tra_num
    tra_num = tra_num+1
    cv2.imwrite('data/tracklets/tracklet_'+str(tra_num)+'.jpg',image)
    cv2.imshow('Tracklet2',image)
    cv2.waitKey(50)

