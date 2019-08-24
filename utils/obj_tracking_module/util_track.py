# cython: profile=True
import cv2
import math
from collections import defaultdict
import string
import random
import numpy as np
import tensorflow as tf
import warnings
from scipy.optimize import linear_sum_assignment

from libc.stdlib cimport malloc, free, realloc
from libc.math cimport round, sqrt
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
# from .vehicle_detection_main cimport Info 
cimport numpy as np

from .appearence_extractor import create_box_encoder

WHITE = (255, 255, 255)
YELLOW = (66, 244, 238)
RED = (0, 10, 255)
GREEN = (80, 220, 60)
LIGHT_CYAN = (255, 255, 224)
DARK_BLUE = (139, 0, 0)
GRAY = (128, 128, 128)

cdef:
    int length = 0
    int counters = 0
    int max_len = 0

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create,
    "goturn":cv2.TrackerGOTURN_create
	}

feature_generator = None
# cdef extern from "opencv2/core/cvstd.hpp" namespace cv:
#     cdef cppclass Ptr:
#         Ptr()
# cdef extern from "opencv2/tracking.hpp" namespace cv:
#     cdef cppclass TrackerKCF:
#         Ptr<TrackerKCF> create()
#         bool init(Mat frame, Rect bbox)
#         bool update(Mat frame, Rect bbox)
cdef struct box:
    int f0
    int f1
    int f2
    int f3
cdef struct Info:
    box bbox
    int age
    int label
    int status

cdef Info *tr = NULL
def load_appearence_model(path_to_model):
    print(path_to_model)
    global feature_generator
    if 'veri' in path_to_model:
        feature_generator = create_box_encoder(path_to_model, batch_size=1)
    else:
        feature_generator = create_box_encoder(path_to_model, input_name = "input_1",
                                output_name = "flatten/Reshape", batch_size=1)


cdef void add_new_object(box obj, np.ndarray image, list trackers, str name, str curr_frame, np.ndarray mask=None) except *:
    cdef:
        int ymin, xmin, ymax, xmax, xmid, ymid
        int age = 0
    # cdef box *initial_bbox
    # label = str(counters["person"]+ counters["car"]+counters["truck"]+ counters["bus"])
    #Age:time for which the tracker is allowed to deviate from its orignal feature 
    # age=0
    # print(obj)
    ymin = <int>obj.f0
    xmin = <int>obj.f1
    ymax = <int>obj.f2
    xmax = <int>obj.f3

    xmid = <int>(round((xmin+xmax)/2))
    ymid = <int>(round((ymin+ymax)/2))
    
    # dist = math.sqrt((center[0] - xmid)**2 + (center[1] - ymid)**2)

    # init tracker
    # tracker = cv2.TrackerKCF_create()  # Note: Try comparing KCF with MIL

    # if dist <= radius*0.93:
    tracker = OPENCV_OBJECT_TRACKERS[name]()
    success = tracker.init(image, (xmin, ymin, xmax-xmin, ymax-ymin))
    if success:
        if mask is not None:
            # try:
            #     tracker.setInitialMask(mask)
            # except Exception as e:
            #     # warnings.warn(str(e))
            #     pass
            feature = feature_generator(image, [(xmin, ymin, xmax-xmin, ymax-ymin)], mask)
            # print(np.asarray(feature).shape)
        else:
            feature = feature_generator(image, [(xmin, ymin, xmax-xmin, ymax-ymin)])
        # print("Adding feature to new track object", np.asarray(feature).shape)
        # initial_bbox = box(xmin, ymin, xmax-xmin, ymax-ymin)
        add_new_Tracker((xmin, ymin, xmax-xmin, ymax-ymin), age, 0)
        
        trackers.append([tracker, feature])
        # print("Car - ", label, "is added")
        # label_object(RED, RED, fontface, image, label, textsize, 4, xmax, xmid, xmin, ymax, ymid, ymin)

cdef bint not_tracked(np.ndarray image, box object_, list trackers, str name, float threshold, str curr_frame_no,
                 str dist_metric, float iou_threshold, np.ndarray mask=None) except -1:
    # print("Eu threshold", threshold)
    # if object_ == (0, 0 ,0 ,0):
    #     # return []  # No new classified objects to search for
    #     return False
    cdef:
        int ymin, xmin, ymax, xmax, ymid, xmid, x1, x2, y1, y2, w, h, age
        int bymin, bxmin, bymax, bxmax, bymid, bxmid, area
        int min_id = -1
        box bbox
        float max_overlap = 0.0, min_dist = 2.0
        float box_range, overlap, dist, eu_dist
        np.ndarray dt_ft, dt_feature
        int active
    global tr
    ymin = <int>object_.f0
    xmin = <int>object_.f1
    ymax = <int>object_.f2
    xmax = <int>object_.f3
    # new_objects = []

    ymid = <int>(round((ymin+ymax)/2))
    xmid = <int>(round((xmin+xmax)/2))

    if not trackers: # use length
        return True

    area = (xmax - xmin + 1) * (ymax - ymin + 1)
    box_range = sqrt((xmax-xmin)**2 + (ymax-ymin)**2)/2    #UNCOMMENT
    for i in range(length):
        (tracker, feature) = trackers[i]
        # print("Not_tracked -- 0 ")
        bbox = tr[i].bbox
        age = tr[i].age
        active = tr[i].status
        car = tr[i].label
        # print("Not_tracked -- 1 ")
        if active == 0 or age <= 3: #less than sampling rate, since inactive trackers can loose out on further immediate det. based on iou 
            bxmin = <int>(bbox.f0)
            bymin = <int>(bbox.f1)
            bxmax = <int>(bbox.f0 + bbox.f2)
            bymax = <int>(bbox.f1 + bbox.f3)
            bxmid = <int>(round((bxmin + bxmax) / 2))
            bymid = <int>(round((bymin + bymax) / 2))
            #IOU-dist
            x1 = max(xmin, bxmin)
            y1 = max(ymin, bymin)
            x2 = min(xmax, bxmax)
            y2 = min(ymax, bymax)

            w = max(0, x2 - x1 + 1)
            h = max(0, y2 - y1 + 1)

            overlap = (w * h)/area
            #Ellipse
            # dist = (((bxmid - xmid)/h_axis)**2 + ((bymid - ymid)/v_axis)**2)
            # print("Not_tracked -- 2 ")

            dist = sqrt((xmid - bxmid)**2 + (ymid - bymid)**2)   #uncomment
            # print("Car no {} is {}units, range is {}".format(car, dist, box_range))
            print("Overlap with Car :",car,i," is", overlap, "Frame", curr_frame_no)
            if dist <= box_range and overlap >= iou_threshold and overlap > max_overlap:
                max_overlap = overlap 
                min_id = i
    if min_id != -1:
        t=trackers[min_id]
        tr[min_id].age=0 #Resetting age on detection
        cv_tr_obj = OPENCV_OBJECT_TRACKERS[name]()
        success = cv_tr_obj.init(image, (xmin, ymin, xmax-xmin, ymax-ymin))
        
        if success:
            with open('./Re-identification.txt', 'a') as f:
                f.write("Updating tracker {} in frame {}\n".format(tr[min_id].label, curr_frame_no))
            # del t[0]
            t[0] = cv_tr_obj             #uncomment 
            dt_feature = feature_generator(image, [(xmin, ymin, xmax-xmin, ymax-ymin)], mask)
            t[1] = np.concatenate((t[1],dt_feature), axis = 0)
            tr[min_id].status = 0
    else:
        # ymin, xmin, ymax, xmax = [int(en) for en in object_]
        dt_ft = feature_generator(image, [(xmin, ymin, xmax-xmin, ymax-ymin)], mask)
        for x in range(length):
            (_, ft) = trackers[x]
            age = tr[x].age

            # a = np.squeeze(np.asarray(ft[-200:]), axis = 1)
            if dist_metric == "cosine":
                eu_dist = _nn_cosine_distance(ft[-200:], dt_ft)
            else:
                eu_dist = _nn_euclidean_distance(ft[-200:], dt_ft)

            # print("car no ", cn, "eu-dist -", eu_dist, "Frame", curr_frame_no, "Age", age)
            if eu_dist < threshold and age > 0 and min_dist > eu_dist:
                # xmin, ymin, xmax, ymax = bx
                min_dist = eu_dist
                min_id = x
        if min_id != -1:
            t =trackers[min_id]
            
            cv_tr_obj = OPENCV_OBJECT_TRACKERS[name]()
            # print((xmin, ymin, xmax-xmin, ymax-ymin))
            success = cv_tr_obj.init(image, (xmin, ymin, xmax-xmin, ymax-ymin))
            
            if success:
                with open('./Re-identification.txt', 'a') as f:
                    f.write("Re-initializing tracker {} age {} status {} in frame {}\n".format(
                        tr[min_id].label,tr[min_id].age, tr[min_id].status, curr_frame_no))
                # print("Re-initializing tracker ",cn, t[2])
                t[0] = cv_tr_obj
                tr[min_id].age = 0
                t[1] = np.concatenate((t[1],dt_ft), axis = 0)
                # t[1].append(dt_ft)
                tr[min_id].status = 0
                # break
        # else:
        #     new_objects.append(object_)
    # print("Not_tracked -- 3 ")    
    return True if min_id == -1 else False


def label_object(color, textcolor, image, car, thickness, xmax, xmid, xmin, ymax, ymid, ymin):
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    # thickness = 1
    textsize, _baseline = cv2.getTextSize(
            str(car), fontface, fontscale, thickness)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
    pos = (xmid - textsize[0]//2, ymid + textsize[1]//2)
    cv2.putText(image, str(car), pos, fontface, 1, textcolor, thickness, cv2.LINE_AA)
    # print("label_object")

def updt_trackers(image, cp_image, trackers, curr_frame, threshold, dist_metric, max_age, sr):
    cdef int car, xmin, ymin, xmax, ymax, xmid, ymid, idx = 0
    cdef box bbox
    # try:
    #     if int(curr_frame)%sr == 0:
    update_trackers(image, cp_image, trackers, curr_frame, threshold, dist_metric, max_age)
    #     else:
    #         while idx < length:
    #             active = tr[idx].status
    #             if active == 0:
    #                 car = tr[idx].label
    #                 bbox = tr[idx].bbox
    #                 xmin = <int>(bbox.f0)
    #                 ymin = <int>(bbox.f1)
    #                 xmax = <int>(bbox.f0 + bbox.f2)
    #                 ymax = <int>(bbox.f1 + bbox.f3)
    #                 xmid = <int>(round((xmin + xmax) / 2))
    #                 ymid = <int>(round((ymin + ymax) / 2))

    #                 label_object(GREEN, RED, image, car, 2, xmax, xmid, xmin, ymax, ymid, ymin)
    #             idx +=1
    # except Exception as e:
    #     print(repr(e))
cdef void update_trackers(np.ndarray image, np.ndarray cp_image, list trackers, str curr_frame, 
                        float threshold, str dist_metric, int max_age=72) except *:
    global length, tr
    color = (80, 220, 60)
    cdef int idx = 0
    cdef int active, age, car
    # cdef bint success 
    cdef int xmin
    cdef int ymin, xmax, ymax, xmid, ymid
    cdef float distance = 2.0
    cdef np.ndarray dt_feature
    # cdef box bbox 
    #2 entities (1) [cv2 tracker instance, features]  (2) [age, status, label, bbox] (a struct called "Info")
    # Traverse both
    while idx < length:
        # tracker, bx, car, age, _, active = trackers[idx]
        # print("length----", length, len(trackers), idx)
        # print("update 1")

        tracker, features = trackers[idx]
        age = tr[idx].age
        car = tr[idx].label
        active = tr[idx].status
        
        # print("update 2")
        # pair = trackers[idx]
        if active == 0:
            success, bbox = tracker.update(image)
        else:
            if age >= max_age:
                # counters['lost_trackers']+=1
                print("Deleting tracker {} with age {} on AOI exit..{}".format(car, age, length))
                del trackers[idx]
                del_Tracker(idx)
                length -= 1
                continue
            idx+=1
            tr[idx].age +=1
            continue
        # print("update 3")
        # print("Tracker object", tracker.update(image))
        if not success:
            tr[idx].status = 1
            print("Deleting tracker", car,"on update failure")
            # print("Lost tracker no.", car)
            # counters['lost_trackers'] += 1
            # del trackers[idx]
            idx+=1
            continue
            
        # print("update 4")
        tr[idx].bbox = box(bbox[0], bbox[1], bbox[2], bbox[3])  #Updating current bbox of tracker "car"
        # print("Age", age)
        # print("length of feats", len(_))
        xmin = <int>bbox[0]
        ymin = <int>bbox[1]
        xmax = <int>(bbox[0] + bbox[2])
        ymax = <int>(bbox[1] + bbox[3])
        xmid = <int>(round((xmin+xmax)/2))
        ymid = <int>(round((ymin+ymax)/2))
        # print("update 5", bbox)
        dt_feature = feature_generator(cp_image, [bbox])
    
        # print("Detection bbox feature shape", np.asarray(dt_feature).shape)
        # a = np.squeeze(np.asarray(features[-200:]), axis = 1)
        # float distance = 2.0
        # print("update 6")
        if dist_metric == "cosine":
            distance = _nn_cosine_distance(features[-200:], dt_feature)
        else:
            distance = _nn_euclidean_distance(features[-200:], dt_feature)
        # print(distance)
        # distance = 2.0
        with open("Cosine-distances.txt", 'a') as f:
            f.write("Tracker no {} : {}, ft_length: {} ,age {}, frame {}, status {}\n".format(
                car, distance, features.shape[0], age, curr_frame, active))
        # print(distance)
        if distance > threshold:
            tr[idx].age +=1
        # print("update 7")
        if age >= max_age:
            # counters['lost_trackers']+=1
            print("Deleting tracker {} with age {} on AOI exit..{}".format(car, age, length))
            del trackers[idx]
            del_Tracker(idx)
            length -= 1
            continue

        # print("update 8")
        label_object(color, RED, image, car, 2, xmax, xmid, xmin, ymax, ymid, ymin)
        idx +=1
    
# def in_range(obj):
#     ymin = obj['ymin']
#     ymax = obj['ymax']
#     if ymin < START_LINE or ymax > ROI_YMAX:
#         # Don't add new trackers before start or after finish.
#         # Start line can help avoid overlaps and tracker loss.
#         # Finish line protection avoids counting the car twice.
#         return False
#     return True

def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)

def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.

    """
    distances = _cosine_distance(x, y)
    # print(distances)
    return distances.min(axis=0)

def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    # a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2

def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    distances = _pdist(x, y)
    # print(distances.shape)
    return np.maximum(0.0, distances.min(axis=0))

def reframe_box_masks_to_image_masks(box_masks, boxes, image_height,
                                     image_width):
  """Transforms the box masks back to full image masks.

  Embeds masks in bounding boxes of larger masks whose shapes correspond to
  image shape.

  Args:
    box_masks: A tf.float32 tensor of size [num_masks, mask_height, mask_width].
    boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
           corners. Row i contains [ymin, xmin, ymax, xmax] of the box
           corresponding to mask i. Note that the box corners are in
           normalized coordinates.
    image_height: Image height. The output mask will have the same height as
                  the image height.
    image_width: Image width. The output mask will have the same width as the
                 image width.

  Returns:
    A tf.float32 tensor of size [num_masks, image_height, image_width].
  """
  # TODO(rathodv): Make this a public function.
  def reframe_box_masks_to_image_masks_default():
    """The default function when there are more than 0 box masks."""
    def transform_boxes_relative_to_boxes(boxes, reference_boxes):
      boxes = tf.reshape(boxes, [-1, 2, 2])
      min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
      max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
      transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
      return tf.reshape(transformed_boxes, [-1, 4])

    box_masks_expanded = tf.expand_dims(box_masks, axis=3)
    num_boxes = tf.shape(box_masks_expanded)[0]
    unit_boxes = tf.concat(
        [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
    reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
    return tf.image.crop_and_resize(
        image=box_masks_expanded,
        boxes=reverse_boxes,
        box_ind=tf.range(num_boxes),
        crop_size=[image_height, image_width],
        extrapolation_value=0.0)
  image_masks = tf.cond(
      tf.shape(box_masks)[0] > 0,
      reframe_box_masks_to_image_masks_default,
      lambda: tf.zeros([0, image_height, image_width, 1], dtype=tf.float32))
  return tf.squeeze(image_masks, axis=3)

def iou_value(box, tracker):
    ymin, xmin, ymax, xmax = box

    ymid = (ymin+ymax)/2
    xmid = (xmin+xmax)/2
    area = (xmax - xmin + 1) * (ymax - ymin + 1)
    box_range = math.sqrt((xmax-xmin)**2 + (ymax-ymin)**2)/2

    bbox = tracker[1]
    bxmin = int(bbox[0])
    bymin = int(bbox[1])
    bxmax = int(bbox[0] + bbox[2])
    bymax = int(bbox[1] + bbox[3])
    bxmid = (bxmin + bxmax) / 2
    bymid = (bymin + bymax) / 2

    dist = math.sqrt((xmid - bxmid)**2 + (ymid - bymid)**2)
    if dist > box_range:
        return 0.0 # No Overlap
    #IOU-dist
    x1 = np.maximum(xmin, bxmin)
    y1 = np.maximum(ymin, bymin)
    x2 = np.minimum(xmax, bxmax)
    y2 = np.minimum(ymax, bymax)

    w = np.maximum(0, x2 - x1 + 1)
    h = np.maximum(0, y2 - y1 + 1)

    overlap = (w * h)/area
    return overlap

def distance_metric_value(image, box, tracker, dist_metric, mask):
    ymin, xmin, ymax, xmax = box
    dt_ft = feature_generator(image, [(xmin, ymin, xmax-xmin, ymax-ymin)], mask)
    ft = tracker[-2]
    a = np.squeeze(np.asarray(ft[-200:]), axis = 1)

    if dist_metric == "cosine":
        eu_dist = _nn_cosine_distance(a, np.asarray(dt_ft))
    else:
        eu_dist = _nn_euclidean_distance(a, np.asarray(dt_ft))
    return eu_dist

def untracked_detections(image, trackers, boxes, name, curr_frame_no, dist_metric,
                         iou_threshold, threshold, masks = None):
    #Create CostMatrix with inverse iou values for linear assignment
    INFY_COST = 100
    #Trackers allowed to match detections based on iou
    allowed_trackers_1 = [i for i, en in enumerate(trackers) if en[-1] or en[3] < 3]
    CT_1 = np.zeros((len(boxes), len(allowed_trackers_1)))

    for i, en in enumerate(boxes):
        for j, tr in enumerate(allowed_trackers_1):
            iv = iou_value(en, trackers[tr])
            if iv < iou_threshold:
                CT_1[i][j] = INFY_COST
            else:
                CT_1[i][j] = 1 / iv

    r1, c1 = linear_sum_assignment(CT_1)

    unmapped_boxes = [i for i, box in enumerate(boxes) if i not in r1]
    mapped_trackers = [allowed_trackers_1[i] for i in c1]
    allowed_trackers_2 = [i for i, en in enumerate(trackers) if en[3] > 0 and i not in mapped_trackers]
    CT_2 = np.zeros((len(unmapped_boxes), len(allowed_trackers_2)))
    for i, en in enumerate(unmapped_boxes):
        for j, tr in enumerate(allowed_trackers_2):
            mask = None if len(masks) == 0 else masks[i]
            dist = distance_metric_value(image, boxes[en] ,trackers[tr], dist_metric, mask)
            if dist > threshold:
                CT_2[i][j] = INFY_COST
            else:
                CT_2[i][j] = dist
            
    r2, c2 = linear_sum_assignment(CT_2)
    
    for idx, en in enumerate(c1):
        t = trackers[allowed_trackers_1[en]]
        t[3] = 0
        ymin, xmin, ymax, xmax = boxes[r1[idx]]
        dt_ft = feature_generator(image, [(xmin, ymin, xmax-xmin, ymax-ymin)], masks[r1[idx]])
        t[4].append(dt_ft)

    for idx, en in enumerate(c2):
        _id = unmapped_boxes[r2[idx]]
        t = trackers[allowed_trackers_2[en]]
        tr = OPENCV_OBJECT_TRACKERS[name]()
        ymin, xmin, ymax, xmax = boxes[_id]
        dt_ft = feature_generator(image, [(xmin, ymin, xmax-xmin, ymax-ymin)], masks[_id])
        success = tr.init(image, (xmin, ymin, xmax-xmin, ymax-ymin))
        if success:
            t[0] = tr
            t[3] = 0
            t[4].append(dt_ft)
            t[-1] = True
    print(len(r1), len(r2), len(c1), len(c2), len(boxes))
    r2 = [unmapped_boxes[i] for i in r2]
    mapped_trackers = set(np.concatenate([r1,r2]).tolist())
    # mapped_trackers = mapped_trackers if mapped_trackers else [] 
    return [(box, masks[i]) for i, box in enumerate(boxes) if i not in mapped_trackers]


cdef Info *add_new_Tracker((int, int, int, int) bbox, int age, int status):
#   cdef Info *tr
    global length, counters, tr, max_len
    length +=1
    counters += 1
    if tr == NULL:
        # print("Memory error")
        tr = <Info *>PyMem_Malloc(sizeof(Info))
        if not tr:
            raise MemoryError()
        tr[0] = Info(box(bbox[0], bbox[1], bbox[2], bbox[3]), age, counters ,status)
    else:
        if max_len < length:
            max_len = length
            tr = <Info *>PyMem_Realloc(tr, (max_len)* sizeof(Info))
            if not tr:
                print("Memory error")
                raise MemoryError()
        tr[length-1] = Info(box(bbox[0], bbox[1], bbox[2], bbox[3]), age, counters ,status)

#   return tr

cdef del_Tracker(int index):
    global tr, length
    for i in range(length - index - 1):
        tr[index+i] = tr[index+i+1]
    # tr[index] = tr[length]
    

