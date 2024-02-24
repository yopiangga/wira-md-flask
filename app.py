import base64
from typing import Type
from urllib import response
from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify, send_file
import os
import time

from io import BytesIO

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import mahotas as mh
import tensorflow as tf
import cv2
import pandas as pd
import pydicom

dir_image = "/home/yopiangga/Documents/Kuliah/PA/riset/upload-file/uploads/jpg/"
dir = ""

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = load_model(dir + 'model/1.1 Model VIT B16 WP 50E Dense 64.h5')

@app.route('/')
@cross_origin()
def home():
    return "home"

@app.route('/prediction', methods=['POST'])
@cross_origin()
def prediction():

    if 'file' not in request.files:
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
    image.save(dir_image + name_image + '.jpg')

    img_width, img_height = 384, 384

    target_size = (img_width, img_height)
    img = image.resize(target_size)

    gray_image = image_to_grayscale(img)

    blur_image = image_to_blur(gray_image)

    threshold_image = image_to_threshold(blur_image)

    morphological_image = image_to_morphological(threshold_image)

    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)  # Add batch dimension
    image_array /= 255.0  
    
    # Get the predicted class index
    predicted_class_index = np.argmax(prediction[0])

    # Map the index to class labels (assuming you have a list of class labels)
    class_labels = ["normal", "hemorrhagic", "ischemic"]  # Replace with your actual class labels
    predicted_class_label = class_labels[predicted_class_index]

    return jsonify(predicted_class_label)

def image_to_grayscale(img):
    gray_image = img.convert("L")
    return gray_image

def image_to_blur(img, kernel_size=(5, 5), sigma=0):
    image = np.array(img)
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    return blurred_image

def image_to_threshold(img, block_size=11):
    image = np.array(img)
    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded_image

def image_to_morphological(img, kernel_size=3):
    image = np.array(img)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened_image


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')