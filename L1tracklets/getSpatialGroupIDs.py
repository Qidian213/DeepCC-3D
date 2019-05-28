import sys
import numpy as np
import scipy
import scipy.cluster.hierarchy as sch 
import matplotlib.pylab as plt
from collections import Counter

### http://bluewhale.cc/2016-04-19/hierarchical-clustering.html
### https://blog.csdn.net/Tan_HandSome/article/details/79371076
### https://blog.csdn.net/Enigma_tong/article/details/79081449

### Perfroms spatial groupping of detections and returns a vector of IDs
#global isch
#isch = 0
def getSpatialGroupIDs(use_groupping,currentDetectionsIDX,pairwiseDistances,track_ops):
    spatialGroupIDs = np.ones(len(currentDetectionsIDX),int)
    if use_groupping:
        agglomeration = sch.linkage(pairwiseDistances,method='average') # Agglomerative hierarchical cluster tree
        numSpatialGroups = round(track_ops['cluster_coeff']*len(currentDetectionsIDX)/track_ops['window_width']) 
        numSpatialGroups = max(numSpatialGroups,1)
        
#        P=sch.dendrogram(agglomeration)
#        global isch 
#        isch = isch+1
#        plt.savefig('plot_dendrogram_'+ str(isch) + '.png')

        while True:
            spatialGroupIDs = sch.fcluster(agglomeration, t=numSpatialGroups,criterion='maxclust')
            count_dict = Counter(spatialGroupIDs)
            max_key = max(count_dict,key = count_dict.get) 
            if count_dict[max_key] <=150:
                return spatialGroupIDs
            numSpatialGroups = numSpatialGroups+1

