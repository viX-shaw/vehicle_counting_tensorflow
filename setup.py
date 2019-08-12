from distutils.core import setup  
from Cython.Build import cythonize  
import numpy
setup(ext_modules = cythonize(["./utils/obj_tracking_module/appearence_extractor.py",
"./utils/obj_tracking_module/util_track.pyx",
 "./utils/obj_tracking_module/util_track.pxd",
 "./utils/visualization_utils.pyx",
 "vehicle_detection_main.pyx"], annotate=True), include_dirs = [numpy.get_include()],
 package_data = {'vehicle_counting_tensorflow/utils/obj_tracking_module': ['*.pxd']})  
