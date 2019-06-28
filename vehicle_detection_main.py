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
parser.add_argument("--threshold", type = float, default= 0.55, required =False)
parser.add_argument("--eu_threshold", type = float, default=0.2, required =False)
parser.add_argument("--age", type = int, default=72, required =False)
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
    masks = None
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            ops = detection_graph.get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            if 'detection_masks:0' in all_tensor_names:
                detection_masks = detection_graph.get_tensor_by_name('detection_masks:0')
                detection_mks = tf.squeeze(detection_masks, [0])
                detection_boxs = tf.squeeze(detection_boxes, [0])
            # for all the frames that are extracted from input video
            while cap.isOpened():
                (ret, frame) = cap.read()

                if not ret:
                    print ('end of the video file...')
                    break

                input_frame = frame
                copy_frame = np.array(Image.fromarray(np.uint8(frame)).copy())

                # if input_frame is copy_frame:
                #     print("Same object")
                # else:
                #     print("Diff objects")
                # input_frame = load_image_into_numpy_array(frame)
                util_track.update_trackers(input_frame, copy_frame, counters, trackers,str(cap.get(1))[:-2], params.age)
                # print("Total trackers ", trackers,"in frame no.", cap.get(1))

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                if cap.get(1) % params.sr == 0:
                    # Actual detection.
                    image_np_expanded = np.expand_dims(copy_frame, axis=0)
                    if 'detection_masks:0' in all_tensor_names:
                        detection_masks_reframed = util_track.reframe_box_masks_to_image_masks(
                                detection_mks, detection_boxs, copy_frame.shape[0], copy_frame.shape[1])
                        
                        detection_masks_reframed = tf.cast(
                            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                        # Follow the convention by adding back the batch dimension
                        detection_masks = tf.expand_dims(
                            detection_masks_reframed, 0)

                        (boxes, scores, classes, num, masks) = \
                            sess.run([detection_boxes, detection_scores,
                                    detection_classes, num_detections, detection_masks],
                                    feed_dict={image_tensor: image_np_expanded})
                        masks = np.squeeze(masks)
                    else:
                        (boxes, scores, classes, num) = \
                            sess.run([detection_boxes, detection_scores,
                                    detection_classes, num_detections],
                                    feed_dict={image_tensor: image_np_expanded})
                        

                # Visualization of the results of a detection.

                    if masks is not None:
                        print(masks.shape, "boxes_shape", boxes.shape)
                    vis_util.visualize_boxes_and_labels_on_image_array(
                    cap.get(1),
                    copy_frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    params.tracker,
                    trackers,
                    counters,
                    instance_masks=masks,
                    use_normalized_coordinates=True,
                    min_score_thresh = params.threshold,
                    eu_threshold = params.eu_threshold,
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
                cv2.putText(input_frame, "Frame -"+str(cap.get(1))[:-2], (1000, 35),
                     font, 0.8, (0,0,255), 2, cv2.FONT_HERSHEY_SIMPLEX)

               
                cv2.imwrite('output_{}/{}.jpg'.format(params.tracker, cap.get(1)), input_frame)
                del frame
                del copy_frame

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
