#cython: language_level=3
cimport numpy as np

cdef visualize_boxes_and_labels_on_image_array(float current_frame_number,
                                                np.ndarray image,
                                                np.ndarray boxes,
                                                np.ndarray classes,
                                                np.ndarray scores,
                                                dict category_index,
                                                str tracker_name,
                                                list trackers,
                                                dict counters,
                                                float boundary,
                                                str metric,
                                                np.ndarray instance_masks=*,
                                                np.ndarray keypoints=*,
                                                bint use_normalized_coordinates=*,
                                                int max_boxes_to_draw=*,
                                                float min_score_thresh=*,
                                                float eu_threshold=*,
                                                float iou_threshold=*,
                                                bint agnostic_mode=*,
                                                int line_thickness=*)