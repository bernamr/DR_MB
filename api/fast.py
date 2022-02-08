from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from io import BytesIO

from DR_MB.predict import load_image, load_model, output

#from DR_MB.predict import load_image, load_model

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


@app.post("/predict")
def predict(img_file: UploadFile = File(...)):

    imagen = load_image(img_file.file)
    output_pred = load_model(imagen)
    prediction = output(output_pred)

    return prediction

    # Load the TFLite model and allocate tensors.
    # interpreter = tf.lite.Interpreter(model_path="model.tflite")
    # interpreter.allocate_tensors()

    # # Get input and output tensors.
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()

    # input_data_type = input_details[0]["dtype"]

    # # Test the model on random input data.
    # imagen = Image.open(img_file.file)
    # imagen = np.asarray(imagen) #pasa aarray



    # input_data = np.resize(imagen, (224, 224, 3))
    # input_data = np.expand_dims(input_data, 0)


    # interpreter.set_tensor(input_details[0]['index'], input_data)
    # interpreter.invoke()

    # # The function `get_tensor()` returns a copy of the tensor data.
    # # Use `tensor()` in order to get a pointer to the tensor.
    # output_data = interpreter.get_tensor(output_details[0]['index'])
    # output_probs = tf.math.softmax(output_data / 255)
    # d = {
    #     "0": np.round(output_probs.numpy()[0][4], 2),
    #     "2": np.round(output_probs.numpy()[0][3], 2),
    #     "2": np.round(output_probs.numpy()[0][1], 2),
    #     "2": np.round(output_probs.numpy()[0][0], 2),
    #     "4": np.round(output_probs.numpy()[0][2], 2)
    # }

    # c = max(d, key=d.get)
