#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 27th January 2018
# ----------------------------------------------

# Imports
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import numpy as np
import csv
import time
import argparse
import copy

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util
from utils.obj_tracking_module import util_track

    
parser = argparse.ArgumentParser(description='Supervised training')
parser.add_argument("--file", type=str, default="sub-1504614469486.mp4", help="video file for inference")
parser.add_argument("--tracker","-t", type = str, default = "kcf", help = "openCV trackers to use\
        csrt: cv2.TrackerCSRT_create,\
        kcf: cv2.TrackerKCF_create,\
        boosting: cv2.TrackerBoosting_create,\
        mil: cv2.TrackerMIL_create,\
        tld: cv2.TrackerTLD_create,\
        medianflow: cv2.TrackerMedianFlow_create,\
        mosse: cv2.TrackerMOSSE_create")
parser.add_argument("--center","-c", type = str, required = False)
parser.add_argument("--radius", type = int, required =False)
parser.add_argument("--sr", type = int, default = 3, required =False, help = "interval at frames are used for detection")

parser.add_argument("--model_name", type = str, default = "ssd_mobilenet_v1_coco_2018_01_28")
        
params = parser.parse_args()
# initialize .csv
with open('traffic_measurement.csv', 'w') as f:
    writer = csv.writer(f)
    csv_line = \
        'Vehicle Type/Size, Vehicle Color, Vehicle Movement Direction, Vehicle Speed (km/h)'
    writer.writerows([csv_line.split(',')])

# if tf.__version__ < '1.4.0':
#     raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!'
#                       )
try:
    if os.path.exists("output_{}".format(params.tracker)) and os.path.isdir("output_{}".format(params.tracker)):
        shutil.rmtree("output_{}".format(params.tracker))
    os.mkdir("output_{}".format(params.tracker))
except Exception as e:
    pass
# input video
cap = cv2.VideoCapture(params.file)

# Variables
total_passed_vehicle = 0  # using it to count vehicles

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
MODEL_NAME = params.model_name
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = \
    'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Download Model
# uncomment if you have not download the model yet
# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

#Initialize the appearence model
util_track.load_appearence_model()

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. Here I use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
        max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width,
            3)).astype(np.uint8)


# Detection
def object_detection_function():
    total_passed_vehicle = 0
    lost_trackers = 0
    trackers = []
    counters = {
    "person": 0,
    "car": 0,
    "truck":0,
    "bus":0,
    "lost_trackers": 0
    }
    speed = 'waiting...'
    direction = 'waiting...'
    size = 'waiting...'
    color = 'waiting...'
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while cap.isOpened():
                (ret, frame) = cap.read()

                if not ret:
                    print ('end of the video file...')
                    break

                input_frame = frame
                copy_frame = np.array(Image.fromarray(np.uint8(frame)).copy())

                if input_frame is copy_frame:
                    print("Same object")
                else:
                    print("Diff objects")
                # input_frame = load_image_into_numpy_array(frame)
                tracker_boxes = util_track.update_trackers(input_frame, counters, trackers,str(cap.get(1))[:-2])
                # print("Total trackers ", trackers,"in frame no.", cap.get(1))

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                if cap.get(1) % params.sr == 0:
                    image_np_expanded = np.expand_dims(input_frame, axis=0)

                    # Actual detection.
                    (boxes, scores, classes, num) = \
                        sess.run([detection_boxes, detection_scores,
                                detection_classes, num_detections],
                                feed_dict={image_tensor: image_np_expanded})

                # Visualization of the results of a detection.
                # (counter, csv_line) = \
                cv2.imwrite("/content/data/{}.jpg".format(cap.get(1)), copy_frame)
                # Smapling frames
                    # counters = \
                    vis_util.visualize_boxes_and_labels_on_image_array(
                    cap.get(1),
                    copy_frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    params.tracker,
                    trackers,
                    tracker_boxes,
                    counters,
                    use_normalized_coordinates=True,
                    line_thickness=4,
                    )

                # if(counter == 1):
                #     print("Detected vehicle in frame no", cap.get(1))
                try:
                    total_passed_vehicle = counters["person"]+counters["car"]+counters["truck"]+counters["bus"]
                    lost_trackers = counters["lost_trackers"]
                except Exception as e:
                    pass

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Detected Vehicles: ' + str(total_passed_vehicle),
                    (10, 35),
                    font,
                    0.8,
                    (255,0,0),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )
                cv2.putText(
                    input_frame,
                    'Lost Trackers: ' + str(lost_trackers),
                    (400, 35),
                    font,
                    0.8,
                    (255,0,0),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )

                # when the vehicle passed over line and counted, make the color of ROI line green
                # if counter == 1:
                #     cv2.line(input_frame, (0, 420), (1280, 420), (0, 0xFF, 0), 5)
                # else:
                #     cv2.line(input_frame, (0, 420), (1280, 420), (0, 0, 0xFF), 5)

                # insert information text to video frame
                # cv2.circle(input_frame, (400, 380), 310, (10,100,210), 2)
                # cv2.rectangle(input_frame, (10, 275), (230, 337), (180, 132, 109), -1)
                # # cv2.putText(
                # #     input_frame,
                # #     'ROI Line',
                # #     (545, 410),
                # #     font,
                # #     0.6,
                # #     (0, 0, 0xFF),
                # #     2,
                # #     cv2.LINE_AA,
                # #     )
                # cv2.putText(
                #     input_frame,
                #     'LAST PASSED VEHICLE INFO',
                #     (11, 290),
                #     font,
                #     0.5,
                #     (0xFF, 0xFF, 0xFF),
                #     1,
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     )
                # cv2.putText(
                #     input_frame,
                #     '-Movement Direction: ' + direction,
                #     (14, 302),
                #     font,
                #     0.4,
                #     (0xFF, 0xFF, 0xFF),
                #     1,
                #     cv2.FONT_HERSHEY_COMPLEX_SMALL,
                #     )
                # cv2.putText(
                #     input_frame,
                #     '-Speed(km/h): ' + speed,
                #     (14, 312),
                #     font,
                #     0.4,
                #     (0xFF, 0xFF, 0xFF),
                #     1,
                #     cv2.FONT_HERSHEY_COMPLEX_SMALL,
                #     )
                # cv2.putText(
                #     input_frame,
                #     '-Color: ' + color,
                #     (14, 322),
                #     font,
                #     0.4,
                #     (0xFF, 0xFF, 0xFF),
                #     1,
                #     cv2.FONT_HERSHEY_COMPLEX_SMALL,
                #     )
                # cv2.putText(
                #     input_frame,
                #     '-Vehicle Size/Type: ' + size,
                #     (14, 332),
                #     font,
                #     0.4,
                #     (0xFF, 0xFF, 0xFF),
                #     1,
                #     cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    # )

                # cv2.imshow('vehicle detection', input_frame)
                cv2.imwrite('output_{}/{}.jpg'.format(params.tracker, cap.get(1)), input_frame)


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if csv_line != 'not_available':
                    with open('traffic_measurement.csv', 'a') as f:
                        writer = csv.writer(f)
                        (size, color, direction, speed) = \
                            csv_line.split(',')
                        writer.writerows([csv_line.split(',')])
            cap.release()
            cv2.destroyAllWindows()


object_detection_function()		
