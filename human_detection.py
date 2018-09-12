# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 13:07:43 2018

@author: jgs808
"""

# In[1]:
import cv2
import datetime

# In[2]:
from utils import detector_utils
detection_graph, sess = detector_utils.load_inference_graph()

# In[3]:

if __name__ == '__main__':
    score_thresh = 0.70
    
    cap = cv2.VideoCapture(0)
    
    num_humans_detect = 2
    
    start_time = datetime.datetime.now()
    
    im_height, im_width = (None, None)
    frame_center = (None)
    
    try:
        while True:
            ret, frame = cap.read()
            # frame = cv2.resize(frame, (320, 240))
            
            if im_height == None:
                im_height, im_width = frame.shape[:2]
                print('Height: ', im_height, ' Width:', im_width)
                
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")
            frame = cv2.resize(frame, (int(im_width), int(im_height)))
            boxes, scores, classes = detector_utils.detect_objects(frame, detection_graph, sess)
            
            print(detector_utils.draw_box_on_image(num_humans_detect, score_thresh, scores, boxes, classes, im_width, im_height, frame))
            #q print(detector_utils.get_object_position(num_humans_detect, score_thresh, scores, boxes, classes, im_width, im_height))                        
            cv2.imshow('Human detector', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            
    except Exception as e:
        print("Error: " + str(e))