import cv2
import math
import numpy as np
import os

def pose2bb(pose,scalingFactor=0):

    renderThreshold = 0.3
    ref_pose = np.array([[0.,   0.], #nose
       [0.,   23.], # neck
       [28.,   23.], # rshoulder
       [39.,   66.], #relbow
       [45.,  108.], #rwrist
       [-28.,   23.], # lshoulder
       [-39.,   66.], #lelbow
       [-45.,  108.], #lwrist
       [20., 106.], #rhip
       [20.,  169.], #rknee
       [20.,  231.], #rankle
       [-20.,  106.], #lhip
       [-20.,  169.], #lknee
       [-20.,  231.], #lankle
       [5.,   -7.], #reye
       [11.,   -8.], #rear
       [-5.,  -7.], #leye
       [-11., -8.], #lear
       ])
       
   
    # Template bounding box   
    ref_bb = np.array([[-50., -15.], #left top
                [50., 240.]])  # right bottom
            
    pose = np.reshape(pose,(18,3))
    valid = np.logical_and(np.logical_and(pose[:,0]!=0,pose[:,1]!=0), pose[:,2] >= renderThreshold)

    if np.sum(valid) < 5:
        bb = np.array([0, 0, 0, 0])
#        print('got an invalid box')
#        print(pose)
        return bb

    points_det = pose[valid,0:2]
    points_reference = ref_pose[valid,:]
    
    # 1a) Compute minimum enclosing rectangle

    base_left = min(points_det[:,0])
    base_top = min(points_det[:,1])
    base_right = max(points_det[:,0])
    base_bottom = max(points_det[:,1])

    # 1b) Fit pose to template
    # Find transformation parameters
    M = points_det.shape[0]
    B = points_det.flatten('F')
    A = np.vstack((np.column_stack((points_reference[:,0], np.zeros((M)), np.ones((M)),  np.zeros((M)))),
         np.column_stack((np.zeros((M)),  points_reference[:,1], np.zeros((M)),  np.ones((M))) )))
    
    
    params = np.linalg.lstsq(A,B)
    params = params[0]
    M = 2
    A2 = np.vstack((  np.column_stack( (ref_bb[:,0], np.zeros((M)), np.ones((M)),  np.zeros((M)))),
         np.column_stack( (np.zeros((M)),  ref_bb[:,1], np.zeros((M)),  np.ones((M)))) ))

    result = np.matmul(A2,params)

    fit_left = min(result[0:2])
    fit_top = min(result[2:4])
    fit_right = max(result[0:2])
    fit_bottom = max(result[2:4])

    # 2. Fuse bounding boxes
    left = min(base_left,fit_left)
    top = min(base_top,fit_top)
    right = max(base_right,fit_right)
    bottom = max(base_bottom,fit_bottom)

    height = bottom - top + 1
    width = right - left + 1

    if(scalingFactor !=0):
        left = left - 0.5*(scalingFactor-1)*width
        top = top - 0.5*(scalingFactor-1)*height
        width = width*scalingFactor
        height = height*scalingFactor
        
    bb = np.array([left, top, width, height])
    return bb

def bbox(pose,scalingFactor=0):
    bboxs = []
    for pose_b in pose[:]:
        bb = pose2bb(pose_b[0:18],scalingFactor)
        bboxs.append(bb)
    return bboxs

def feet_position(boxes):
    
    x = boxes[0] + 0.5*boxes[2];
    y = boxes[1] + boxes[3];
    feet = np.array([x, y]);
    return feet
    
def getBoundingBoxCenters(bbox):
    center = [[box[2]+0.5*box[4],box[3]+0.5*box[5]] for box in bbox]
    return center
    
def get_bb_image(img, bb,camera_size):
    bb = np.round(bb)
    if bb[2] < 20 or bb[3] < 20:
        return np.zeros((256,128,3))
            
    left = np.maximum(0,bb[0]).astype('int')
    right = np.minimum(camera_size[0]-1,bb[0]+bb[2]).astype('int')
    top = np.maximum(0,bb[1]).astype('int')
    bottom = np.minimum(camera_size[1]-1,bb[1]+bb[3]).astype('int')
    if left == right or top == bottom:
    	return np.zeros((256,128,3))
    snapshot = img[top:bottom,left:right,:]
    snapshot = cv2.resize(snapshot,(128, 256))
    return snapshot
    
def convert_img(img):
    img = img.astype('float')
    img = img / 255.0
    img = img - 0.5
    return img

print("test----line")
