from distutils.core import setup  
from Cython.Build import cythonize  
import numpy
setup(ext_modules = cythonize(["./utils/obj_tracking_module/appearence_extractor.py",
"./utils/obj_tracking_module/util_track.pyx",
 "./utils/obj_tracking_module/util_track.pxd",
 "./utils/visualization_utils.pyx",
 "vehicle_detection_main.pyx"], annotate=True), include_dirs = [numpy.get_include()],
 include_path = ["utils/obj_tracking_module"],
 package_data = {'utils': ['obj_tracking_module/*.pxd']})  
