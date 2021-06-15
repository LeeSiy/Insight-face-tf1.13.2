import os
import time

import cv2
import numpy as np
from keras.applications.imagenet_utils import preprocess_input

from nets.retinaface import RetinaFace
from utils.anchors import Anchors
from utils.config import cfg_mnet, cfg_re50
from utils.utils import BBoxUtility, letterbox_image, retinaface_correct_boxes
from retinaface import RetinaFace
from align_trans import get_reference_facial_points, warp_and_crop_face
from PIL import Image

class Retinaface(object):
    _defaults = {
        "model_path"        : 'path/to/retinaface_resnet50.h5',
        "backbone"          : 'resnet50',
        "confidence"        : 0.75,
        "nms_iou"           : 0.45,
        "input_shape"       : [224, 224, 3],
        "letterbox_image"   : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        if self.backbone == "mobilenet":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50
        self.bbox_util = BBoxUtility(nms_thresh=self.nms_iou)
        self.generate()
        self.anchors = Anchors(self.cfg, image_size=(self.input_shape[0], self.input_shape[1])).get_anchors()

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        self.retinaface = RetinaFace(self.cfg, self.backbone)
        self.retinaface.load_weights(self.model_path,by_name=True)

        print('{} model, anchors, and classes loaded.'.format(model_path))

    def box_image(self, image):
        old_image = image.copy()

        image = np.array(image, np.float32)
        im_height, im_width, _ = np.shape(image)

        scale = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]]
        scale_for_landmarks = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0]]

        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
       
        photo = np.expand_dims(preprocess_input(image),0)
        
        preds = self.retinaface.predict(photo)

        results = self.bbox_util.detection_out(preds, self.anchors, confidence_threshold=self.confidence)

        if len(results)<=0:
            return old_image

        results = np.array(results)

        if self.letterbox_image:
            results = retinaface_correct_boxes(results, np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))
        
        results[:,:4] = results[:,:4]*scale
        results[:,5:] = results[:,5:]*scale_for_landmarks
        scale = 112 / 112.
        reference = get_reference_facial_points(default_square = True) * scale
        face_count = 1
        for b in results:
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            landmarks = [[b[5], b[6]],[b[7], b[8]],[b[9], b[10]],[b[11], b[12]],[b[13], b[14]]]
            landmarks = np.array(landmarks)
            
            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(old_image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            cv2.putText(old_image, "detected{}".format(face_count), (cx, cy-10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            side_predict = (b[5]-b[7])/(b[6]-b[12])
            if side_predict < 0.78:
                cv2.putText(old_image, "Side face", (cx, cy-20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            else: 
                cv2.putText(old_image, "Front face", (cx, cy-20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            
            face_count += 1
            print(b[0], b[1], b[2], b[3], b[4])
            
            cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)
            
        return old_image

    def detect_image(self, image):
        old_image = image.copy()

        image = np.array(image, np.float32)
        im_height, im_width, _ = np.shape(image)

        scale = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]]
        scale_for_landmarks = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0]]

        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
       
        photo = np.expand_dims(preprocess_input(image),0)
        
        preds = self.retinaface.predict(photo)

        results = self.bbox_util.detection_out(preds, self.anchors, confidence_threshold=self.confidence)

        if len(results)<=0:
            return old_image

        results = np.array(results)

        if self.letterbox_image:
            results = retinaface_correct_boxes(results, np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))
        
        results[:,:4] = results[:,:4]*scale
        results[:,5:] = results[:,5:]*scale_for_landmarks
        scale = 112 / 112.
        reference = get_reference_facial_points(default_square = True) * scale
        face_count = 0
        for b in results:
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            landmarks = [[b[5], b[6]],[b[7], b[8]],[b[9], b[10]],[b[11], b[12]],[b[13], b[14]]]
            landmarks = np.array(landmarks)
            warped_face = warp_and_crop_face(np.array(old_image), landmarks, reference, crop_size=(112, 112))
            warped_face = np.expand_dims(warped_face, axis=0)
            if face_count == 0:
                warped_faces = warped_face
                face_count+=1
            else:
                warped_faces = np.concatenate((warped_faces, warped_face), axis = 0)
                face_count+=1
            
        return warped_faces

    def get_FPS(self, image, test_interval):
        image = np.array(image, np.float32)
        im_height, im_width, _ = np.shape(image)

        scale = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]]
        scale_for_landmarks = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0]]

        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
            
        photo = np.expand_dims(preprocess_input(image),0)
        preds = self.retinaface.predict(photo)
        results = self.bbox_util.detection_out(preds, self.anchors, confidence_threshold=self.confidence)

        if len(results)>0:
            results = np.array(results)

            if self.letterbox_image:
                results = retinaface_correct_boxes(results, np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))
        
            results[:,:4] = results[:,:4]*scale
            results[:,5:] = results[:,5:]*scale_for_landmarks
            
        t1 = time.time()
        for _ in range(test_interval):
            preds = self.retinaface.predict(photo)
            results = self.bbox_util.detection_out(preds, self.anchors, confidence_threshold=self.confidence)

            if len(results)>0:
                results = np.array(results)

                if self.letterbox_image:
                    results = retinaface_correct_boxes(results, np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))
                
                results[:,:4] = results[:,:4]*scale
                results[:,5:] = results[:,5:]*scale_for_landmarks
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
