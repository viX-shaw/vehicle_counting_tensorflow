from distutils.core import setup  
from Cython.Build import cythonize
import numpy   
setup(ext_modules = cythonize(["./utils/obj_tracking_module/appearence_extractor.py",
"./utils/obj_tracking_module/util_track.pyx", "./utils/visualization_utils.py"],
 annotate=True),  
include_dirs=[numpy.get_include()])
