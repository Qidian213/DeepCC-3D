import numpy as np

def mergeResults(result1, result2):
    ## Merges the independent solutions for the two results
    labels_1 = result1['labels']
    labels_1 = np.array(labels_1)
    if(len(labels_1)==0):
        maximumLabel = 0.0
    else:
        maximumLabel = np.max(labels_1)
    
    mergeResult = {}
    mergeResult['labels'] = []
    mergeResult['observations'] = []
    
    mergeResult['labels'].extend(result1['labels'])
    mergeResult['labels'].extend(maximumLabel+result2['labels'])
    
    mergeResult['observations'].extend(result1['observations'])
    mergeResult['observations'].extend(result2['observations'])

    return mergeResult
