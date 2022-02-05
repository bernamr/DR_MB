import json
import streamlit as st
import requests
from PIL import Image
import tensorflow as tf
import numpy as np

file = st.file_uploader("Upload Images")
if file:

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.

    imagen = Image.open(file)
    imagen = np.asanyarray(imagen)
    input_data = np.resize(imagen, (224, 224,3))
    input_data = np.expand_dims(input_data, 0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_probs = tf.math.softmax(output_data / 255)
    output_pred = np.array(output_probs[0])

    st.image(imagen)

    nivel = {
        0: 'grado 3',
        1: 'grado 2',
        2: 'grado 4',
        3: 'grado 1',
        4: 'grado 0'
    }

    for i in nivel.keys():
        if np.where(output_pred == output_pred.max())[0][0] == i:
            grado = nivel[i]

    st.write(f'**{grado}**')


else:
    print('no es una imagen')
    params =None
