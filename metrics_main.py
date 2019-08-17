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
import time

from collections import defaultdict, OrderedDict
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
parser.add_argument("--use_masks", type = int, default=0, required =False)
parser.add_argument("--iou_threshold", type = float, default=0.7, required =False)
parser.add_argument("--boundary", type = float, default=80.0, required =False)
parser.add_argument("--metric", type = str, default="cosine", required =False)
parser.add_argument("--feat_model", type = str, default="/content/veri.pb", required = False)

parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)

params = parser.parse_args()
try:
    if os.path.exists("output_{}".format(params.tracker)) and os.path.isdir("output_{}".format(params.tracker)):
        shutil.rmtree("output_{}".format(params.tracker))
    os.mkdir("output_{}".format(params.tracker))
except Exception as e:
    pass

# Variables
total_passed_vehicle = 0  # using it to count vehicles

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

#Initialize the appearence model
util_track.load_appearence_model(params.feat_model)

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. Here I use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
        max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Helper code

def get_detboxes_classes_and_scores(detection_mat, frame_idx):
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detections=[]
    scores = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        detections.append(bbox)
        scores.append(confidence)
        # detection_list.append(Detection(bbox, confidence, feature))
    classes = np.ones((len(scores),), dtype = int)  # For person according to mscoco
    return np.asarray(detections), np.asarray(scores), classes

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
    
            # for all the frames that are extracted from input video
    start_time = time.time()
    image_dir = os.path.join(params.sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(params.sequence_dir, "gt/gt.txt")

    detections = np.loadtxt(os.path.join(params.sequence_dir, "det/det.txt")
                                                        , delimiter=',')
    print(image_filenames)
    for key, entry in OrderedDict(sorted(image_filenames.items())).items():
        boxes ,scores, classes = get_detboxes_classes_and_scores(detections, key)
        # (ret, frame) = cap.read()
        input_frame = np.uint8(np.asarray(Image.open(entry)))
        copy_frame = input_frame.copy()
        # if not ret:
        #     print ('end of the video file...')
        #     break

        # input_frame = frame
        # copy_frame = np.array(Image.fromarray(np.uint8(frame)).copy())

        # if input_frame is copy_frame:
        #     print("Same object")
        # else:
        #     print("Diff objects")
        # input_frame = load_image_into_numpy_array(frame)
        util_track.update_trackers(input_frame, copy_frame, counters, trackers, str(key), params.eu_threshold, params.metric, params.age)
        # print("Total trackers ", trackers,"in frame no.", cap.get(1))

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3] 

        # Visualization of the results of a detection.

            # if masks is not None:
                # print(masks.shape, "boxes_shape", boxes.shape)
            # vis_util.add_or_match_detections_to_trackers(    
        vis_util.visualize_boxes_and_labels_on_image_array(
        str(key),
        copy_frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        params.tracker,
        trackers,
        counters,
        params.boundary,
        params.metric,
        params.sequence_dir,
        instance_masks=masks,
        use_normalized_coordinates=True,
        min_score_thresh = params.threshold,
        eu_threshold = params.eu_threshold,
        iou_threshold = params.iou_threshold,
        line_thickness=4,
        )
        t2 = time.time()
        time_taken = float(t2 -t1)
        sys.stdout.flush()
        # print("\rFPS -", str(1.0/time_taken), end='')
        print("\rFPS - {}".format(str(1.0/time_taken)), end = '')


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
        # del detection_masks

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # if csv_line != 'not_available':
        #     with open('traffic_measurement.csv', 'a') as f:
        #         writer = csv.writer(f)
        #         (size, color, direction, speed) = \
        #             csv_line.split(',')
        #         writer.writerows([csv_line.split(',')])
    end_time = time.time()
    print("Total time --", float(start_time - end_time))
    cap.release()
    cv2.destroyAllWindows()

object_detection_function()		