import time
import tensorflow as tf
import cv2
import numpy as np
from retinaface import RetinaFace
from align_trans import get_reference_facial_points, warp_and_crop_face
from PIL import Image
from retinaface import Retinaface
import os
from numpy.linalg import norm

def predict(img):
    retinaface = Retinaface()
    mode = "predict"
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    result_crop = np.zeros((1,112,112,3))
    result_box = np.zeros((112,112,3))
    if mode == "predict":

        count_img = 0

        img = img
        image = cv2.imread(img)
        if image is None:
            print('Open Error! Try again!')
            return result_crop, result_box
        else:
            image   = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            r_image = retinaface.detect_image(image)
            face_crop = np.zeros((1,112,112,3))
            for i, img in enumerate(r_image):
                img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                img = np.expand_dims(img, axis=0)
                if i == 0:
                    face_crop = img
                else:
                    face_crop = np.concatenate((face_crop,img), axis = 0)
            box_image = retinaface.box_image(image)
            result_box = cv2.cvtColor(box_image,cv2.COLOR_RGB2BGR)
            return face_crop, result_box

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        while(True):
            t1 = time.time()
            ref,frame=capture.read()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = np.array(retinaface.detect_image(frame))
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                    
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()
        return result_crop, result_box
    elif mode == "fps":
        test_interval = 100
        img = cv2.imread('img/street.jpg')
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        tact_time = retinaface.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video' or 'fps'.")
        return result_crop, result_box
