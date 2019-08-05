from distutils.core import setup  
from Cython.Build import cythonize  
setup(ext_modules = cythonize(["./obj_tracking_module/appearence_extractor.py",
"./obj_tracking_module/util_track.py", "vizualization_utils.py"], annotate=True))  