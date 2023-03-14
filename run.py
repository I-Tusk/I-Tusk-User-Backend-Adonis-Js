import os                                                                  # To Perform OS level works.
import six
import cv2                                                                 # OpenCV for Computer Vision                                                        # It has a dictionary that contains colors for each label
import argparse                                                            # To get arguments
import collections
import numpy as np
import pyttsx3                                                             # To perform text to speech function
import threading                                                           # To perform multi-threading operations
import playsound                                                           # To play sounds
import tensorflow as tf                                                    # Main Library.
from object_detection.utils import label_map_util                          # To handle label map.
from object_detection.utils import config_util                             # To load model pipeline.
from object_detection.utils import visualization_utils as viz_utils        # To draw rectangles.
from object_detection.builders import model_builder 