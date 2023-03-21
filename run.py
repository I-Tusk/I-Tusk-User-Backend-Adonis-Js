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

#Text to speech setup.
engine = pyttsx3.init()
en_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"  # female
ru_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_RU-RU_IRINA_11.0"  # male
engine.setProperty('voice', en_voice_id)
rate = engine.getProperty('rate')
engine.setProperty('rate', rate - 25)

#Comment 3