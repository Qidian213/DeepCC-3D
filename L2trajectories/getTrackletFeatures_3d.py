import numpy as np

def getTrackletFeatures_3d(tracklets):
    centersWorld = [] ### [[X,Y,Z,T],[X,Y,Z,T]....]
    intervals = []   ### [[frame_0,frame_n],[frame_0,frame_n]....]
    duration = []   ### [frame_n-frame_0,frame_n-frame_0,....]

    startpoint_3d = [] #### [[x,y,z],[x,y,z]....]
    endpoint_3d = []   #### [[x,y,z],[x,y,z]....]
    velocity_3d = []   ### [[vx,vy,vz],[vx,vy,vz],....]
    
    numTracklets = len(tracklets)
    
    ## bounding box centers for each tracklets
    for i in range(numTracklets):
        detections = tracklets[i]['data']  ### [frame, id, x, y, w, h, x_3d, y_3d, z_3d]
        
        # 3d points
        bboxes = detections  ### 
        tmp_bb = [[bbx[6],bbx[7],bbx[8],bbx[0]] for bbx in bboxes]
        centersWorld.append(tmp_bb)
        
     ## calculate velocity, direction, for each tracklet
    for ind in range(numTracklets):
        intervals.append([centersWorld[ind][0][3],centersWorld[ind][-1][3]])
        startpoint_3d.append([centersWorld[ind][0][0],centersWorld[ind][0][1],centersWorld[ind][0][2]])
        endpoint_3d.append([centersWorld[ind][-1][0],centersWorld[ind][-1][1],centersWorld[ind][-1][2]])
        
        duration.append(centersWorld[ind][-1][3]-centersWorld[ind][0][3])
        direction_3d = [endpoint_3d[ind][0]-startpoint_3d[ind][0],endpoint_3d[ind][1]-startpoint_3d[ind][1],endpoint_3d[ind][2]-startpoint_3d[ind][2]]
        velocity_3d.append([direction_3d[0]/duration[ind],direction_3d[1]/duration[ind],direction_3d[2]/duration[ind]])
        
    return startpoint_3d,endpoint_3d, intervals, duration, velocity_3d
