import json
import streamlit as st
import requests
from PIL import Image
import tensorflow as tf
import numpy as np
from DR_MB.predict import load_image, load_model, output



url = 'http://127.0.0.1:8000/predict'
file = st.file_uploader("Upload an image", type=["jpeg", 'jpg', 'png'])
img = {'img_file': file}
if file:

    st.image(file)
    api = requests.post(url, files=img)
    response = api.json()




    #prediction = output(output_pred)

    st.write(f'**{response}**')

else:
    print('no es una imagen')
    params =None
