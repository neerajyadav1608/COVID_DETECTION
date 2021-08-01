from flask import Flask, render_template, request,jsonify
from keras.models import load_model
import cv2
import numpy as np
import base64
from PIL import Image
import io
import re

img_size=100

app = Flask(__name__,static_url_path='') 

model_CT=load_model('CT_Covid_best_model_2.hdf5')

label={0:'Covid19 Positive', 1:'Covid19 Negative'}

def preprocess_CT(img):
    img_size=100
    img=np.array(img)

    if(img.ndim==3):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray=img

    gray=gray/255
    resized=cv2.resize(gray,(img_size,img_size))
    reshaped=resized.reshape(1,img_size,img_size)
    return reshaped
@app.route("/css")
def css():
	return app.send_static_file("./templates/CSS/ss.css")
@app.route("/")
def index():
	return(render_template("index.html"))

@app.route("/predict", methods=["POST"])
def predict():
            print('HERE')
            message = request.get_json(force=True)
            encoded = message['image']
            decoded = base64.b64decode(encoded)
            dataBytesIO=io.BytesIO(decoded)
            dataBytesIO.seek(0)
            image = Image.open(dataBytesIO)

            test_image=preprocess_CT(image)
            prediction1 = model_CT.predict(test_image)
            result1=np.argmax(prediction1,axis=1)[0]
            Probality1=float(np.max(prediction1,axis=1)[0])
            result1=label[result1]
            print(prediction1,result1,Probality1)

            response = {'prediction': {'result': result1,'accuracy': Probality1}}

            return jsonify(response)

app.run(debug=True)

#<img src="" id="img" crossorigin="anonymous" width="400" alt="Image preview...">
