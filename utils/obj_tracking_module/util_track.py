import cv2
import math
from collections import defaultdict
import string
import random
import numpy as np
import tensorflow as tf

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

def add_new_object(obj, image, counters, trackers, name, curr_frame, mask=None):
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
        if mask is not None:
            tracker.setInitialMask(mask)
            feature = feature_generator(image, [(xmin, ymin, xmax-xmin, ymax-ymin)], mask)
        else:
            feature = feature_generator(image, [(xmin, ymin, xmax-xmin, ymax-ymin)])
        # print("Adding feature to new track object", np.asarray(feature).shape)
        trackers.append([tracker, (xmin, ymin, xmax-xmin, ymax-ymin), label, age, [feature]])
        # print("Car - ", label, "is added")
    # label_object(RED, RED, fontface, image, label, textsize, 4, xmax, xmid, xmin, ymax, ymid, ymin)

def not_tracked(image, object_, trackers, threshold, curr_frame_no, mask=None):
    # print("Eu threshold", threshold)
    if not object_:
        # return []  # No new classified objects to search for
        return False

    ymin, xmin, ymax, xmax = object_
    new_objects = []

    ymid = int(round((ymin+ymax)/2))
    xmid = int(round((xmin+xmax)/2))

    # dist = math.sqrt((center[0] - xmid)**2 + (center[1] - ymid)**2)
    # if dist<=radius*0.93:
    if not trackers:
        # return objects  # No existing boxes, return all objects
        return True

    box_range = math.sqrt((xmax-xmin)**2 + (ymax-ymin)**2)/2    #UNCOMMENT
    # box_range = 10.0

    for i, (tracker, bbox, car_no, _, feature) in enumerate(trackers):
        bxmin = int(bbox[0])
        bymin = int(bbox[1])
        bxmax = int(bbox[0] + bbox[2])
        bymax = int(bbox[1] + bbox[3])
        bxmid = int((bxmin + bxmax) / 2)
        bymid = int((bymin + bymax) / 2)
        dist = math.sqrt((xmid - bxmid)**2 + (ymid - bymid)**2)   #uncomment
        # print("Car no {} is {}units, range is {}".format(car_no, dist, box_range))
        if dist <= box_range:
            # print("car no ", car_no, "is in range")
            # found existing, so break (do not add to new_objects)
            #compute cosine distance b/w track feature and matched detection

            #in the parameters also pass features of all tracks
            dt_feature = feature_generator(image, [(xmin, ymin, xmax-xmin, ymax-ymin)], mask)
            # print("Detection bbox feature shape", np.asarray(dt_feature).shape)
            # distance = _nn_cosine_distance(np.asarray(feature), np.asarray(dt_feature))
            # with open("Cosine-distances.txt", 'a') as f:
            #     f.write("Tracker no {} : {}\n".format(i, distance))

            # if distance > 2.2:
            #     #needs the whole track object
            #     del trackers[i]
            t=trackers[i]
            t[3]=0 #Resetting age on detection
            t[4].append(dt_feature) # at 118
            if dist <= 7.0: #15.0 
                tr = OPENCV_OBJECT_TRACKERS["csrt"]()
                success = tr.init(image, (xmin, ymin, xmax-xmin, ymax-ymin))
                if mask is not None:
                    tr.setInitialMask(mask)
                if success:
                    with open('./Re-identification.txt', 'a') as f:
                        f.write("Updating tracker {} in frame {}\n".format(car_no, curr_frame_no))
                # del t[0]
                t[0] = tr             #uncomment 
            break
    else:
        ymin, xmin, ymax, xmax = [int(en) for en in object_]
        dt_ft = feature_generator(image, [(xmin, ymin, xmax-xmin, ymax-ymin)], mask)
        for x, (_, _, cn, _, ft) in enumerate(trackers):

            a = np.squeeze(np.asarray(ft[-72:]), axis = 1)

            eu_dist = _nn_euclidean_distance(a, np.asarray(dt_ft))
            # print("car no ", cn, "eu-dist -", eu_dist)
            if eu_dist < threshold:
                # xmin, ymin, xmax, ymax = bx
                t =trackers[x]

                tr = OPENCV_OBJECT_TRACKERS["csrt"]()
                # print((xmin, ymin, xmax-xmin, ymax-ymin))
                success = tr.init(image, (xmin, ymin, xmax-xmin, ymax-ymin))
                if mask is not None:
                    tr.setInitialMask(mask)
                if success:
                    with open('./Re-identification.txt', 'a') as f:
                        f.write("Re-initializing tracker {} in frame {}\n".format(cn, curr_frame_no))
                    # print("Re-initializing tracker ",cn, t[2])
                    # del t[0]
                    t[0] = tr
                    t[3] = 0
                    t[4].append(dt_ft)
                    break
        else:
            new_objects.append(object_)

    return True if len(new_objects)>0 else False


def label_object(color, textcolor, fontface, image, car, textsize, thickness, xmax, xmid, xmin, ymax, ymid, ymin):
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
    pos = (xmid - textsize[0]//2, ymid + textsize[1]//2)
    cv2.putText(image, car, pos, fontface, 1, textcolor, thickness, cv2.LINE_AA)


def update_trackers(image, cp_image, counters, trackers, curr_frame, max_age=72):
    # print("Max age", max_age)
    color = (80, 220, 60)
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    thickness = 1
    idx = 0

    # for n, pair in enumerate(trackers):
    # print("Trackers ",[t[1] for t in trackers])
    while idx < len(trackers):
        tracker, bx, car, age, _ = trackers[idx]
        textsize, _baseline = cv2.getTextSize(
            car, fontface, fontscale, thickness)
        success, bbox = tracker.update(image)
        # print("Tracker object", tracker.update(image))
        pair = trackers[idx]
        if not success:
            print("Deleting tracker", car,"on update failure")
            counters['lost_trackers'] += 1
            # print("Lost tracker no.", car)
            del trackers[idx]
            continue

        pair[1] = bbox  #Updating current bbox of tracker "car"
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
            pair[3]+=1
        # else:
        #     pair[3].append(dt_feature)

        if age >= max_age:
            print("Deleting tracker {} with age {} on AOI exit..".format(car, age))
            del trackers[idx]
            continue

        # boxes.append((bbox, car, _))  # Return updated box list        

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
        box_indices=tf.range(num_boxes),
        crop_size=[image_height, image_width],
        extrapolation_value=0.0)
  image_masks = tf.cond(
      tf.shape(box_masks)[0] > 0,
      reframe_box_masks_to_image_masks_default,
      lambda: tf.zeros([0, image_height, image_width, 1], dtype=tf.float32))
  return tf.squeeze(image_masks, axis=3)