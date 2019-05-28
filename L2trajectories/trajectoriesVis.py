import cv2
import numpy as np
import matplotlib.pyplot as plt  
import random

def trajectoriesVis(Trajectories):
    image = cv2.imread("data/vis.jpg")
    for ind in range(len(Trajectories)):
        trajectory = Trajectories[ind]
        tracklets = trajectory['tracklets']
        color = [random.randint(0,255) for i in range(3)]
        for intra in range(len(tracklets)):
            centers = tracklets[intra]['data']
            for bbx in centers:
                cv2.circle(image, (int(bbx[2]+0.5*bbx[4]),int(bbx[3]+0.5*bbx[5])), 1, color, 4)
                
    cv2.imwrite('data/trajectories/vis.jpg',image)
    cv2.imshow('Tracklet',image)
    cv2.waitKey(100)
    
