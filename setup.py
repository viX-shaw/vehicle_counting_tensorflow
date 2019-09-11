from distutils.core import setup, Extension  
from Cython.Build import cythonize  
import numpy

setup(ext_modules = cythonize(["./utils/obj_tracking_module/util_track.pyx",
 "./utils/obj_tracking_module/util_track.pxd",
 "./utils/visualization_utils.pyx"], annotate=True, gdb_debug = True), include_dirs = [numpy.get_include()],
 package_data = {'vehicle_counting_tensorflow/utils/obj_tracking_module': ['*.pxd']},
 library_dirs = ['/content/E/vehicle_counting_tensorflow/utils/obj_tracking_module'],  
 libraries = ['KCF', 'cvt'])

# _DEBUG_LEVEL = 0
# extra_compile_args = ["-g3", "-O0", "-DDEBUG=%s" % _DEBUG_LEVEL, "-UNDEBUG"]
# exttension = Extension("util_track",["./utils/obj_tracking_module/util_track.pyx"],
#                 extra_compile_args=extra_compile_args)
# setup(ext_modules = cythonize([exttension,
#  "./utils/visualization_utils.pyx"], annotate=True), include_dirs = [numpy.get_include()],
#  package_data = {'vehicle_counting_tensorflow/utils/obj_tracking_module': ['*.pxd']})  