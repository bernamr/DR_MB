from email.mime import image
from tkinter import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import cv2
import numpy as np
from DR_MB.predict import load_image, load_model


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict")
async def predict(file): #Como entra el input
    #extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    #if not extension:
    #  print ("Image must be jpg or png format!")

    return {file: 'prueba ok'}


    # Load the TFLite model and allocate tensors.
    #interpreter = tf.lite.Interpreter(model_path="model.tflite")
    #interpreter.allocate_tensors()

    # Get input and output tensors.
    #input_details = interpreter.get_input_details()
    #output_details = interpreter.get_output_details()

    # Test the model on random input data.
    #imagen = cv2.imread(file.read(), cv2.IMREAD_GRAYSCALE)
    #input_data = cv2.resize(imagen, (224, 224), interpolation=cv2.INTER_AREA)
    #input_data = np.expand_dims(input_data, 0)
    #interpreter.set_tensor(input_details[0]['index'], input_data)
    #interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    #output_data = interpreter.get_tensor(output_details[0]['index'])
    #output_probs = tf.math.softmax(output_data / 255)
    #output_pred = np.array(output_probs[0])

    #return 'hola'
    return type(img)
