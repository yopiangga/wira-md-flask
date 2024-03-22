from typing import Type
from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify, send_file
import os
import time

from io import BytesIO

from tensorflow.keras.models import load_model

from PIL import Image
import numpy as np
import tensorflow as tf
import pydicom

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)
CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'

dir_self = os.environ.get("DIR_SELF", default="true")
dir_express = os.environ.get("DIR_EXPRESS", default="true")

model_path = dir_self + "model/model_vit"
jpg_path = dir_express + "uploads/medical-record/jpg/"
segmented_path = dir_express + "uploads/medical-record/segmented/"

model = load_model(model_path)

@app.route('/')
@cross_origin()
def home():
    return "home"

@app.route('/prediction', methods=['POST'])
@cross_origin()
def prediction():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    file_data = file.read()

    file_object = BytesIO(file_data)
    im = pydicom.dcmread(file_object)
    im = im.pixel_array.astype(float)

    rescaled_image = (np.maximum(im,0)/im.max()) * 255
    image = np.uint8(rescaled_image)
    image = Image.fromarray(image)
    image = image.convert("RGB")
    nameImage = str(time.time())
    image.save(nameImage + '.jpg')

    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.keras.preprocessing.image.smart_resize(image_array, (384, 384))
    image_array = tf.expand_dims(image_array, 0)
    image_array /= 255.0

    prediction = model.predict(image_array)

    predicted_class_index = np.argmax(prediction[0])

    class_labels = ["normal", "hemorrhagic", "ischemic"]
    predicted_class_label = class_labels[predicted_class_index]

    return jsonify(predicted_class_label)

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
