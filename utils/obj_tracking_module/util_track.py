import cv2
import math
from collections import defaultdict
import string
import random
import numpy as np

from .appearence_extractor import create_box_encoder

WHITE = (255, 255, 255)
YELLOW = (66, 244, 238)
RED = (0, 10, 255)
GREEN = (80, 220, 60)
LIGHT_CYAN = (255, 255, 224)
DARK_BLUE = (139, 0, 0)
GRAY = (128, 128, 128)

radius = 310
center = (400, 380)

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

def load_appearence_model():
    global feature_generator
    feature_generator = create_box_encoder("/content/veri.pb", batch_size=1)

def add_new_object(obj, image, counters, trackers, name, curr_frame):
    ymin, xmin, ymax, xmax = obj
    label = str(counters["person"]+ counters["car"]+counters["truck"]+ counters["bus"])
    #Age:time for which the tracker is allowed to deviate from its orignal feature 
    age=0
    # print(obj)
    ymin = int(ymin)
    xmin = int(xmin)
    ymax = int(ymax)
    xmax = int(xmax)

    xmid = int(round((xmin+xmax)/2))
    ymid = int(round((ymin+ymax)/2))
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    thickness = 1
    textsize, _baseline = cv2.getTextSize(
        label, fontface, fontscale, thickness)

    dist = math.sqrt((center[0] - xmid)**2 + (center[1] - ymid)**2)

    # init tracker
    # tracker = cv2.TrackerKCF_create()  # Note: Try comparing KCF with MIL

    # if dist <= radius*0.93:
    tracker = OPENCV_OBJECT_TRACKERS[name]()
    success = tracker.init(image, (xmin, ymin, xmax-xmin, ymax-ymin))
    if success:
        feature = feature_generator(image, [(xmin, ymin, xmax-xmin, ymax-ymin)])
        # print("Adding feature to new track object", np.asarray(feature).shape)
        trackers.append([tracker, label, age, [feature]])
        print("Car - ", label, "is added")
    # label_object(RED, RED, fontface, image, label, textsize, 4, xmax, xmid, xmin, ymax, ymid, ymin)

def not_tracked(image, object_, boxes, trackers):
    if not object_:
        # return []  # No new classified objects to search for
        return False

    ymin, xmin, ymax, xmax = object_
    new_objects = []
    f = 1
    ymid = int(round((ymin+ymax)/2))
    xmid = int(round((xmin+xmax)/2))

    dist = math.sqrt((center[0] - xmid)**2 + (center[1] - ymid)**2)
    # if dist<=radius*0.93:
    if not boxes:
        # return objects  # No existing boxes, return all objects
        return True
    box_range = ((xmax - xmin) + (ymax - ymin)) / 2
    for i, (bbox, car_no, feature) in enumerate(boxes):
        bxmin = int(bbox[0])
        bymin = int(bbox[1])
        bxmax = int(bbox[0] + bbox[2])
        bymax = int(bbox[1] + bbox[3])
        bxmid = int((bxmin + bxmax) / 2)
        bymid = int((bymin + bymax) / 2)
        dist = math.sqrt((xmid - bxmid)**2 + (ymid - bymid)**2)
        # print("Car no {} is {}units, range is {}".format(car_no, dist, box_range))
        if dist <= box_range:
            # print("car no ", car_no, "is in range")
            # found existing, so break (do not add to new_objects)
            #compute cosine distance b/w track feature and matched detection

            #in the parameters also pass features of all tracks
            dt_feature = feature_generator(image, [bbox])
            # print("Detection bbox feature shape", np.asarray(dt_feature).shape)
            # distance = _nn_cosine_distance(np.asarray(feature), np.asarray(dt_feature))
            # with open("Cosine-distances.txt", 'a') as f:
            #     f.write("Tracker no {} : {}\n".format(i, distance))

            # if distance > 2.2:
            #     #needs the whole track object
            #     del trackers[i]
            t=trackers[i]
            t[2]=0 #Resetting age on detection
            t[3].append(dt_feature)
            f=0
            break
    if f==1:
        new_objects.append(object_)

    return True if len(new_objects)>0 else False


def label_object(color, textcolor, fontface, image, car, textsize, thickness, xmax, xmid, xmin, ymax, ymid, ymin):
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
    pos = (xmid - textsize[0]//2, ymid + textsize[1]//2)
    cv2.putText(image, car, pos, fontface, 1, textcolor, thickness, cv2.LINE_AA)


def update_trackers(image, cp_image, counters, trackers, curr_frame):
    boxes = []
    color = (80, 220, 60)
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    thickness = 1
    idx = 0

    # for n, pair in enumerate(trackers):
    while idx < len(trackers):
        tracker, car, age, _ = trackers[idx]
        textsize, _baseline = cv2.getTextSize(
            car, fontface, fontscale, thickness)
        success, bbox = tracker.update(image)
        # print("Tracker object", tracker.update(image))
        pair = trackers[idx]
        if not success:
            counters['lost_trackers'] += 1
            # print("Lost tracker no.", car)
            del trackers[n]
            continue

        # print("Age", age)
        # print("length of feats", len(_))
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[0] + bbox[2])
        ymax = int(bbox[1] + bbox[3])
        xmid = int(round((xmin+xmax)/2))
        ymid = int(round((ymin+ymax)/2))

        dt_feature = feature_generator(cp_image, [bbox])
    
        # print("Detection bbox feature shape", np.asarray(dt_feature).shape)
        a = np.squeeze(np.asarray(_[-72:]), axis = 1)
        distance = _nn_euclidean_distance(a, np.asarray(dt_feature))
        # print(distance)
        with open("Cosine-distances.txt", 'a') as f:
            f.write("Tracker no {} : {}, ft_length: {} ,age {}\n".format(car, distance, len(_), age))
        # print(distance)
        if abs(distance) > 0.2:
            # print("Working")
            #needs the whole track object
            pair[2]+=1
        # else:
        #     pair[3].append(dt_feature)

        if age >= 30:
            print("Deleting tracker {} with age {} on AOI exit..".format(car, age))
            del trackers[idx]
            continue

        boxes.append((bbox, car, _))  # Return updated box list        

        # if ymid >= ROI_YMAX:
        #     label_object(WHITE, WHITE, fontface, image, car, textsize, 1, xmax, xmid, xmin, ymax, ymid, ymin)
        #     # Count left-lane, right-lane as cars ymid crosses finish line
        #     if xmid < 630:
        #         left_lane += 1
        #     else:
        #         right_lane += 1
        #     # Stop tracking cars when they hit finish line
        #     del trackers[n]
        # else:
            # Rectangle and number on the cars we are tracking
        label_object(color, RED, fontface, image, car, textsize, 2, xmax, xmid, xmin, ymax, ymid, ymin)
        idx +=1
    # Add finish line overlay/line
    # overlay = image.copy()

    # # Shade region of interest (ROI). We're really just using the top line.
    # cv2.rectangle(overlay,
    #               (0, ROI_YMAX),
    #               (FRAME_WIDTH, FRAME_HEIGHT), DARK_BLUE, cv2.FILLED)
    # cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # # Draw start line, if > 0
    # if START_LINE > 0:
    #     cv2.line(image, (0, START_LINE), (FRAME_WIDTH, START_LINE), GRAY, 4, cv2.LINE_AA)
    # # Draw finish line with lane hash marks
    # cv2.line(image, (0, ROI_YMAX), (FRAME_WIDTH, ROI_YMAX), LIGHT_CYAN, 4, cv2.LINE_AA)
    # cv2.line(image, (350, ROI_YMAX - 20), (350, ROI_YMAX + 20), LIGHT_CYAN, 4, cv2.LINE_AA)
    # cv2.line(image, (630, ROI_YMAX - 20), (630, ROI_YMAX + 20), LIGHT_CYAN, 4, cv2.LINE_AA)
    # cv2.line(image, (950, ROI_YMAX - 20), (950, ROI_YMAX + 20), LIGHT_CYAN, 4, cv2.LINE_AA)

    # # Add lane counter
    # cv2.putText(image, "Lane counter:", (30, ROI_YMAX + 80), fontface, 1.5, LIGHT_CYAN, 4, cv2.LINE_AA)
    # cv2.putText(image, str(left_lane), (480, ROI_YMAX + 80), fontface, 1.5, LIGHT_CYAN, 4, cv2.LINE_AA)
    # cv2.putText(image, str(right_lane), (800, ROI_YMAX + 80), fontface, 1.5, LIGHT_CYAN, 4, cv2.LINE_AA)
    # seconds = counters['frames'] / FRAME_FPS
    # cv2.putText(image, "Cars/second:", (35, ROI_YMAX + 110), fontface, 0.5, LIGHT_CYAN, 1, cv2.LINE_AA)
    # cv2.putText(image, '{0:.2f}'.format(left_lane / seconds), (480, ROI_YMAX + 110), fontface, 0.5, LIGHT_CYAN, 1, cv2.LINE_AA)
    # cv2.putText(image, '{0:.2f}'.format(right_lane / seconds), (800, ROI_YMAX + 110), fontface, 0.5, LIGHT_CYAN, 1, cv2.LINE_AA)

    # counters['left_lane'] = left_lane
    # counters['right_lane'] = right_lane
    return boxes

def in_range(obj):
    ymin = obj['ymin']
    ymax = obj['ymax']
    if ymin < START_LINE or ymax > ROI_YMAX:
        # Don't add new trackers before start or after finish.
        # Start line can help avoid overlaps and tracker loss.
        # Finish line protection avoids counting the car twice.
        return False
    return True

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

