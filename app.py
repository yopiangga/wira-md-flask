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
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)
CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'

# dir_self = os.environ.get("DIR_SELF")
# dir_express = os.environ.get("DIR_EXPRESS")

dir_self = "/home/yopiangga/Documents/Kuliah/Lomba/51 KMIPN/code/wira-md-flask/"
dir_express = "/home/yopiangga/Documents/Kuliah/Lomba/51 KMIPN/code/wira-md-express/"

# model_path = str(dir_self) + "model/1.1 Best Model VIT_0.9622_0.9837"
model_path = str(dir_self) + "model/2.1 Best Model VIT_0.9689_0.8592"
model_segmentation_path = str(dir_self) + "model/model_unet_segmentation_9"

jpg_path = str(dir_express) + "uploads/medical-record/jpg/"
segmented_path = str(dir_express) + "uploads/medical-record/segmented/"

model = load_model(model_path)
model_segmentation = load_model(model_segmentation_path)

@app.route('/')
@cross_origin()
def home():
    return dir_self

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
    nameImage = request.form.get('name')
    image.save(jpg_path + nameImage + '.jpg')

    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.keras.preprocessing.image.smart_resize(image_array, (384, 384))
    image_array = tf.expand_dims(image_array, 0)
    image_array /= 255.0

    prediction = model.predict(image_array)

    prediction_segmentation = model_segmentation.predict(image_array)
    image_segmented = prediction_segmentation[0]

    image_segmented = image_segmented * 255
    cv2.imwrite(segmented_path + nameImage + '.jpg', image_segmented)

    predicted_class_index = np.argmax(prediction[0])

    class_labels = ["ischemic", "hemorrhagic", "normal"]
    predicted_class_label = class_labels[predicted_class_index]

    return jsonify(predicted_class_label)

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
