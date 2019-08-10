cimport numpy as np
cdef struct Info:
  (int, int, int ,int) bbox
  int age
  int label
  bint status

cpdef update_trackers(np.ndarray image, np.ndarray cp_image, Info *tr, trackers, str curr_frame, float threshold, str dist_metric, int max_age=*)

cpdef not_tracked(np.ndarray image, int[:] object_, Info *tr_info, trackers, str name, float threshold, str curr_frame_no,str dist_metric, float iou_threshold, np.ndarray mask=*)
cpdef add_new_object(int[:] obj, np.ndarray image,Info *tr, trackers, str name, str curr_frame, np.ndarray mask=*)

