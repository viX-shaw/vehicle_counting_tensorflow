cimport numpy as np
cdef struct Info:
  (int, int, int ,int) bbox
  int age
  int label
  bint status

cdef update_trackers(np.ndarray image, np.ndarray cp_image, Info *tr, list trackers, str curr_frame, float threshold, str dist_metric, int max_age=*)

cdef not_tracked(np.ndarray image, (int, int, int, int) object_, Info *tr_info, list trackers, str name, float threshold, str curr_frame_no,str dist_metric, float iou_threshold, np.ndarray mask=*)
cdef add_new_object((int, int, int, int) obj, np.ndarray image,Info *tr, list trackers, str name, str curr_frame, np.ndarray mask=*)

