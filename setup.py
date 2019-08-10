from distutils.core import setup  
from Cython.Build import cythonize  
import numpy
setup(ext_modules = cythonize(["./utils/obj_tracking_module/appearence_extractor.py",
"./utils/obj_tracking_module/util_track.pyx",
 "./utils/obj_tracking_module/util_track.pxd",
 "./utils/visualization_utils.py",
 "vehicle_detection_main.py"], annotate=True), include_dirs = [numpy.get_include()])  
