# Load Image
image = st.file_uploader("Choose an image")

if st.button("Make Prediction"):
    if image is not None:
        params = {"img_file": image.getvalue()}
        api_url = "http://127.0.0.1:8000/predict"
        res = requests.get(api_url, files=params)
        pred = res.json()



@app.get("/predict")
def predict(img_file: UploadFile = File(...)):
    model_path = "model_model-export_icn_tflite-dr_deep_model_v2_20220126044745-2022-01-29T21_41_28.325372Z_model.tflite"
    labels = [1, 3, 2, 0]
    model = TensorflowLiteClassificationModel(model_path, labels)
    input_image = Image.open(img_file.file)
    label_probability = model.run_from_filepath(np.array((input_image)))
    return label_probability
