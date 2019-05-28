import numpy as np

def getTrackletFeatures(tracklets):
    centersWorld = [] ### [[X,Y,T],[X,Y,T]....]
    centersView = [] ### [[X,Y,T],[X,Y,T]....]
    startpoint = []  ### [[x,y],[x,y]....]
    endpoint = []    ### [[x,y],[x,y]....]
    intervals = []   ### [[frame_0,frame_n],[frame_0,frame_n]....]
    duration = []   ### [frame_n-frame_0,frame_n-frame_0,....]
    velocity = []   ### [[vx,vy],[vx,vy],....]
    
    numTracklets = len(tracklets)
    
    ## bounding box centers for each tracklets
    for i in range(numTracklets):
        detections = tracklets[i]['data']
        
        # 2d points
        bboxes = detections  ### 
        tmp_bb = [[bbx[2]+0.5*bbx[4],bbx[3]+0.5*bbx[5],bbx[0]] for bbx in bboxes]
        centersView.append(tmp_bb)
        centersWorld.append(tmp_bb)
        
     ## calculate velocity, direction, for each tracklet
    for ind in range(numTracklets):
        intervals.append([centersWorld[ind][0][2],centersWorld[ind][-1][2]])
        startpoint.append([centersWorld[ind][0][0],centersWorld[ind][0][1]])
        endpoint.append([centersWorld[ind][-1][0],centersWorld[ind][-1][1]])
        
        duration.append(centersWorld[ind][-1][2]-centersWorld[ind][0][2])
        direction = [endpoint[ind][0]-startpoint[ind][0],endpoint[ind][1]-startpoint[ind][1]]
        velocity.append([direction[0]/duration[ind],direction[1]/duration[ind]])
        
#    print(centersWorld[0])
#    print(startpoint[0])
#    print(endpoint[0])
#    print(duration[0])
#    print(velocity[0])
#    print('tracklets number: ',numTracklets)
#    print(len(intervals),len(intervals[0]))
#    print(len(startpoint),len(startpoint[0]))
#    print(len(endpoint),len(endpoint[0]))
#    print(len(duration))
#    print(len(direction))
#    print(len(velocity),len(velocity[0]))

    return  centersWorld, centersView, startpoint, endpoint, intervals, duration, velocity
