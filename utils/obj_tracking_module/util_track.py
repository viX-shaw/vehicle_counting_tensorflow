import cv2
import math
from collections import defaultdict

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

prev_tracker_update = defaultdict()

def add_new_object(obj, image, counters, trackers, name, curr_frame):
    ymin, xmin, ymax, xmax = obj
    label = str(counters["person"]+ counters["car"]+counters["truck"]+ counters["bus"])

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

    if dist <= radius*0.93:
        tracker = OPENCV_OBJECT_TRACKERS[name]()
        success = tracker.init(image, (xmin, ymin, xmax-xmin, ymax-ymin))
        prev_tracker_update[label] = (ymin, xmin, ymax, xmax)
        if success:
            trackers.append((tracker, label, curr_frame))
        label_object(RED, RED, fontface, image, label, textsize, 4, xmax, xmid, xmin, ymax, ymid, ymin)

def not_tracked(object_, boxes):
    if not object_:
        # return []  # No new classified objects to search for
        return False

    ymin, xmin, ymax, xmax = object_
    new_objects = []
    
    ymid = int(round((ymin+ymax)/2))
    xmid = int(round((xmin+xmax)/2))

    dist = math.sqrt((center[0] - xmid)**2 + (center[1] - ymid)**2)
    if dist<=radius*0.93:
        if not boxes:
            # return objects  # No existing boxes, return all objects
            return True
        box_range = ((xmax - xmin) + (ymax - ymin)) / 2
        for bbox in boxes:
            bxmin = int(bbox[0])
            bymin = int(bbox[1])
            bxmax = int(bbox[0] + bbox[2])
            bymax = int(bbox[1] + bbox[3])
            bxmid = int((bxmin + bxmax) / 2)
            bymid = int((bymin + bymax) / 2)
            if math.sqrt((xmid - bxmid)**2 + (ymid - bymid)**2) < box_range:
                # found existing, so break (do not add to new_objects)
                break
        else:
            new_objects.append(object_)

    return True if len(new_objects)>0 else False


def label_object(color, textcolor, fontface, image, car, textsize, thickness, xmax, xmid, xmin, ymax, ymid, ymin):
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
    pos = (xmid - textsize[0]//2, ymid + textsize[1]//2)
    cv2.putText(image, car, pos, fontface, 1, textcolor, thickness, cv2.LINE_AA)


def update_trackers(image, counters, trackers, curr_frame):
    boxes = []
    color = (80, 220, 60)
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    thickness = 1

    for n, pair in enumerate(trackers):
        tracker, car, frame = pair
        age = int(curr_frame) - int(frame) 
        textsize, _baseline = cv2.getTextSize(
            car, fontface, fontscale, thickness)
        # success, bbox = tracker.update(image)
        print("Tracker object", tracker.update(image))

        if not success:
            counters['lost_trackers'] += 1
            # print("Lost tracker no.", car)
            del trackers[n]
            continue

        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[0] + bbox[2])
        ymax = int(bbox[1] + bbox[3])
        xmid = int(round((xmin+xmax)/2))
        ymid = int(round((ymin+ymax)/2))

        # p_ymin, p_xmin, p_ymax, p_xmax = prev_tracker_update[car]
        # p_xmid = 610
        # p_ymid = 380

        dist = math.sqrt((center[0] - xmid)**2 + (center[1] - ymid)**2)
        with open('details.txt', 'a') as f:
            f.write( "{} moved {} units from centre\n".format(car, dist))
        # print("Tracker no", car, "moved", dist, "units")
        # prev_tracker_update[car] = (ymin, xmin, ymax, xmax)

        if dist > radius or age >= 180:
            print("Deleting tracker {} with age {} on AOI exit..".format(car, age))
            del trackers[n]
            continue

        boxes.append(bbox)  # Return updated box list        

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
        label_object(color, RED, fontface, image, car, textsize, 4, xmax, xmid, xmin, ymax, ymid, ymin)

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