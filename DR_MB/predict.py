import re
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image


def load_image(image):
    imagen = Image.open(image)
    imagen = np.asanyarray(imagen)
    input_data = np.resize(imagen, (224, 224, 3))
    input_data = np.expand_dims(input_data, 0)

    return input_data

def load_model(image):
    #loda model
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

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

def output(output):
    labels = {
        0: 'grado 3',
        1: 'grado 2',
        2: 'grado 4',
        3: 'grado 1',
        4: 'grado 0'
    }

    for i in labels.keys():
        if np.where(output == output.max())[0][0] == i:
            grado = labels[i]

    return grado
