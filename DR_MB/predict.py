import re
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image


def load_image(image):
    imagen = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    input_data = cv2.resize(imagen, (224, 224), interpolation=cv2.INTER_AREA)
    input_data = np.expand_dims(input_data, 0)

    return input_data

def load_model(image):
    #loda model
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    interpreter = load_model()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set tensor
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_probs = tf.math.softmax(output_data / 255)
    output_pred = np.array(output_probs[0])

    return output_pred
