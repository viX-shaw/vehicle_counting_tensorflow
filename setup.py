from distutils.core import setup  
from Cython.Build import cythonize  
setup(ext_modules = cythonize(["./utils/obj_tracking_module/appearence_extractor.py",
"./utils/obj_tracking_module/util_track.pyx"], annotate=True))  
