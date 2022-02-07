import json
import streamlit as st
import requests
from PIL import Image
import tensorflow as tf
import numpy as np
from DR_MB.predict import load_image, load_model, output




file = st.file_uploader("Upload Images")
#api = requests.get('http://127.0.0.1:8000')
#response = api.json()
if file:
    imagen = load_image(file)


    output_pred = load_model(imagen)

    prediction = output(output_pred)

    st.write(f'**{prediction}**')

else:
    print('no es una imagen')
    params =None
