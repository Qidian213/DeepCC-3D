import cv2
import numpy as np
import matplotlib.pyplot as plt  
import random

#def trackletsVis(spatialGroupDetectionCenters,labels):
#    labels = np.ravel(labels)
#    ids = np.unique(labels)
#    fig = plt.figure() 
#    ax1 = fig.add_subplot(111)
#    ax1.set_title('Scatter Plot')  
#    plt.xlabel('X')
#    plt.ylabel('Y') 
#    plt.ylim(0,1080) 
#    plt.xlim(0,1920) 
#    cValue = ['r','y','g','b','r','y','g','b','r'] 
#    for idr in ids:
#        mask = np.where(labels == idr) 
#        centeres_id = spatialGroupDetectionCenters[mask]
#        x = [val[0] for val in centeres_id]
#        y = [val[1] for val in centeres_id]
#        if len(x)<20:
#            continue
#        ax1.scatter(x,y,c=cValue[int(idr)],marker='s')  
#    plt.show()  

def trackletsVis(spatialGroupDetectionCenters,labels):
    labels = np.ravel(labels)
    ids = np.unique(labels)
    image = cv2.imread("L1tracklets/vis.jpg")

    for idr in ids:
        color = [random.randint(0,255) for cl in range(3)]
        mask = np.where(labels == idr) 
        centeres_id = spatialGroupDetectionCenters[mask]
        if(len(centeres_id))<20 :
            continue

        for val in centeres_id:
            cv2.circle(image, (int(val[0]),int(val[1])), 1, color, 4)
            
    cv2.imwrite('L1tracklets/vis.jpg',image)
    cv2.imshow('Tracklet',image)
    cv2.waitKey(50)

