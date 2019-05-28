import cv2
import numpy as np
import matplotlib.pyplot as plt  
import random

def trajectoriesVis_det(detections):
    detections = np.array(detections)
    personIDs = np.unique(detections[:,1])
    
    for i in range(len(personIDs)):
        
        image = cv2.imread("data/vis.jpg")
        color = [random.randint(0,255) for cl in range(3)]
        
        personID = personIDs[i]
        person_det = np.where(detections[:,1]==personID)
        relevantDetections = detections[person_det]
        for det in relevantDetections:
            if(det[2]==0 and det[3]==0):
                continue
            cv2.circle(image, (int(det[2]+0.5*det[4]),int(det[3]+0.5*det[5])), 1, color, 4)
            
        cv2.imwrite('data/trajectories/traj_'+str(i)+'.jpg',image)


#    image = cv2.imread("data/vis.jpg")
#    for i in range(len(personIDs)):
#        color = [random.randint(0,255) for cl in range(3)]
#        
#        personID = personIDs[i]
#        person_det = np.where(detections[:,1]==personID)
#        relevantDetections = detections[person_det]
#        for det in relevantDetections:
#            if(det[2]==0 and det[3]==0):
#                continue
#            cv2.circle(image, (int(det[2]+0.5*det[4]),int(det[3]+0.5*det[5])), 1, color, 4)
#            
#    cv2.imwrite('data/trajectories/vis1.jpg',image)

