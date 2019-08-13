from distutils.core import setup  
from Cython.Build import cythonize  
import numpy
setup(ext_modules = cythonize(["./utils/obj_tracking_module/util_track.pyx",
 "./utils/obj_tracking_module/util_track.pxd",
 "./utils/visualization_utils.pyx"], annotate=True), include_dirs = [numpy.get_include()],
 package_data = {'vehicle_counting_tensorflow/utils/obj_tracking_module': ['*.pxd']})  
