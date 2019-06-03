import cv2
from timeit import time
import h5py
import numpy as np
from PIL import Image

##openpose
from external import Deepose
from external import utils

##ReID network
from external import ReID_feat

def main():
    ## openpose
    Pose = Deepose.DeepPose()
    
    # create features encoder
    image_encoder = ReID_feat.ImageEncoder('external/resnet50_model_130.pth')
    
    video_capture = cv2.VideoCapture('/home/zzg/Datasets/Duke/dukevideos/camera5/00002.MTS')
    w = int(video_capture.get(3))
    h = int(video_capture.get(4))
    camera_size = (w,h)
    
    #h5 file to store final embeddings
    f_out = h5py.File('data/features.h5', 'w')
    emb_storage = []
    frame_num =0
    cam = 1 
    
    fps = 0.0
    while True:
        ret, frame = video_capture.read()
        if ret != True:
            break;
        t1 = time.time()
        keypoints,output_image = Pose.getKeypoints(frame)
        bboxs = utils.bbox(keypoints,1.25)
        if len(bboxs) != 0:
            for bb in bboxs:
                cv2.rectangle(output_image,(int(bb[0]), int(bb[1])), (int(bb[0]+bb[2]), int(bb[1]+bb[3])),(255,255,255), 2)
            cv2.imshow(" ",output_image);
            features = image_encoder.encoder(frame,bboxs,camera_size)
            for bbox, feature in zip(bboxs, features):
              #  print(feature.size())
                fe_bb = np.zeros(2057, np.float32)  # [cam,frame,x,y,w,h,dx,dy,dz,feature]
                fe_bb[0] = cam
                fe_bb[1] = frame_num
                fe_bb[2:6] = bbox
                fe_bb[6] = frame_num
                fe_bb[7] = bbox[0]
                fe_bb[8] = bbox[1]
               # print(bbox)
                fe_bb[9:] = feature
                emb_storage.append(fe_bb)
        frame_num = frame_num+1
        
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("frame_num= %d   fps= %f"%(frame_num,fps))
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    
    # Store the final embeddings.
    emb_dataset = f_out.create_dataset('emb', data=emb_storage)
    video_capture.release()
    cv2.destroyAllWindows()
    f_out.close()
    
if __name__ == '__main__':
    main()
    
