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

#Comment 3
def talk_function(text):               # Text to speech convertion
    print("Computer: {}".format(text))
    engine.say(text)
    engine.runAndWait()

    

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model_config_path =  f'data/models/ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config'        # Store the path of config file
checkpoint_model_path   =  f'data/models/ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/ckpt-0'      # Store the path of model
label_map_path    =  f'data/mscoco_label_map.pbtxt'                             # Store the path of label_map 


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(model_config_path)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(checkpoint_model_path).expect_partial()

@tf.function

def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

    category_index = label_map_util.create_category_index_from_labelmap(label_map_path,
                                                                    use_display_name=True)

    cap = cv2.VideoCapture("test-video.m4v")