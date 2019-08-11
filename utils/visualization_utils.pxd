cimport numpy as np
from utils.obj_tracking_module.util_track cimport Info

cpdef visualize_boxes_and_labels_on_image_array(float current_frame_number,
                                                np.ndarray image,
                                                np.ndarray boxes,
                                                np.ndarray classes,
                                                np.ndarray scores,
                                                np.ndarray category_index,
                                                str tracker_name,
                                                Info *tr_info,
                                                list trackers,
                                                dict counters,
                                                float boundary,
                                                str metric,
                                                np.ndarray instance_masks=None,
                                                np.ndarray keypoints=None,
                                                bint use_normalized_coordinates=False,
                                                int max_boxes_to_draw=40,
                                                float min_score_thresh=.55,
                                                float eu_threshold=0.2,
                                                float iou_threshold=0.7,
                                                bint agnostic_mode=False,
                                                int line_thickness=4):