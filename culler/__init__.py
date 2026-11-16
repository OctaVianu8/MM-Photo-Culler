"""MM Photo Culler - local culling for concert / low-light burst photography."""
import os

# Quiet the native libraries before anything imports them.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

__version__ = "0.1.0"
