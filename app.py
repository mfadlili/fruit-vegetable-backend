from flask import Flask, jsonify, request
import numpy as np
import base64
import json
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import cv2

app = Flask(__name__)

dict_class = {0: 'Fresh Apple',
 1: 'Fresh Banana',
 2: 'Fresh Bitter Gourd',
 3: 'Fresh Capsicum',
 4: 'Fresh Orange',
 5: 'Fresh Tomato',
 6: 'Stale Apple',
 7: 'Stale Banana',
 8: 'Stale Bitter Gourd',
 9: 'Stale Capsicum',
 10: 'Stale Orange',
 11: 'Stale Tomato'}

def json2im(jstr):
    """Convert a JSON string back to a Numpy array"""
    load = json.loads(jstr)
    imdata = base64.b64decode(load['image'])
    im = pickle.loads(imdata)
    return im

@app.route('/')
def home():
    return "Helloworld"

@app.route('/fruit', methods=['GET', 'POST'])
def fruit():
    if request.method=="POST":
        data = request.json
        json_to_img = json2im(data)
        res = np.array([json_to_img[:,:,2].T, json_to_img[:,:,1].T, json_to_img[:,:,0].T]).T
        path = ''
        cv2.imwrite(os.path.join(path , 'inf_image.jpg'), np.array(res))
        img = image.load_img('inf_image.jpg', target_size=(128,128))
        x = image.img_to_array(img)
        model = load_model("fruit_veg_model.h5")
        classes = np.argmax(model.predict(np.array([x])/255))
        response = {'code':200, 'status':'OK', 
                    'result':dict_class[int(classes)]}
        return jsonify(response)
    return "Gunakan method post"

