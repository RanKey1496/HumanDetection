# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 13:23:47 2018

@author: jgs808
"""

# In[1]:
import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
import cv2
from collections import defaultdict
from object_detection.utils import label_map_util

# In[2]:

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'

PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# In[3]:
def load_inference_graph():
    print("--- Loading frozen graph into memory ---")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    return detection_graph, sess

# In[4]:
def draw_box_on_image(num_humans, score_tresh, scores, boxes, classes, im_width, im_height, image_np):
    color = (0, 50, 255)
    frame_center = (int(im_width/2), int(im_height/2))
    rp1, rp2 = get_rectangle_points(im_width, im_height)
    cv2.rectangle(image_np, rp1, rp2, (125, 28, 51), 3, 1)
    cv2.circle(image_np, frame_center, 1, (0, 255, 0), -1)
    positions = []
    for i in range(num_humans):
        if (scores[i] > score_tresh):
            if classes[i] == 1:
                (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                                boxes[i][0] * im_height, boxes[i][2] * im_height)
                p1 = (int(left), int(top))
                p2 = (int(right), int(bottom))
                
                cv2.rectangle(image_np, p1, p2, color, 3, 1)
                cv2.putText(image_np, 'Phone', (int(left), int(top)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(image_np, 'Prob: ' + str("{0:.2f}".format(scores[i])), (int(left), int(top)-20),
                                                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                positions.append((left, right, top, bottom))
    return positions
                
def get_rectangle_points(im_width, im_height):
    (left, top, right, bottom) = (im_width/4, im_height/4, im_width*(3/4), im_height*(3/4))
    return (int(left), int(top)), (int(right), int(bottom))   
 
# Solo devuelve el valor del primer humano encontrado
def get_object_position(num_humans, score_tresh, scores, boxes, classes, im_width, im_height):
    for i in range(num_humans):
        if (scores[i] > score_tresh):
            if (classes[i] == 1):
                return (boxes[i][1] * im_width, boxes[i][3] * im_width, boxes[i][0] * im_height,
                        boxes[i][2] * im_height)            
                
# In[5]:
def draw_text_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
    
# In[6]:
def detect_objects(image_np, detection_graph, sess):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    (boxes, scores, classes, num) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)

# In[7]:
