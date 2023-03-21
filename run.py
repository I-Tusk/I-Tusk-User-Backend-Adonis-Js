# To Perform OS level works.
import os
import six
# OpenCV for Computer Vision                                                        # It has a dictionary that contains colors for each label
import cv2
# To get arguments
import argparse
import collections
import numpy as np
# To perform text to speech function
import pyttsx3
# To perform multi-threading operations
import threading
# To play sounds
import playsound
# Main Library.
import tensorflow as tf
# To handle label map.
from object_detection.utils import label_map_util
# To load model pipeline.
from object_detection.utils import config_util
# To draw rectangles.
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

# Text to speech setup.
engine = pyttsx3.init()
en_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"  # female
ru_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_RU-RU_IRINA_11.0"  # male
engine.setProperty('voice', en_voice_id)
rate = engine.getProperty('rate')
engine.setProperty('rate', rate - 25)

# Comment 3

# comment 4
