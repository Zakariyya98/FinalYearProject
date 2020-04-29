from __future__ import division, print_function
import json
# Imported Modules
import pandas as pd
import numpy as np
import biosppy
import matplotlib.pyplot as plt
import re
import sys
import os
import glob
import cv2
# Keras Libraries
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
# I will be using Flask to direct this to a web app
from flask import Flask, redirect, url_for, request, render_template
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename

# Flask WebApp
app = Flask(__name__)

# This is where I will load my trained Arrhythmia Model
model = load_model('models/ECGArrhythmiaModel.hdf5')
model.summary()
model._make_predict_function()
print('Arrhythmia Model Active..')
output = []

def Arrhythmia_Model(file_upload, model):
    flag = 1
    
    for path in file_upload:
        APC, NORMAL = [], []
        output.append(str(path))
        result = {"APC": APC, "Normal": NORMAL}

        
        indices = []
        
        kernel = np.ones((4,4),np.uint8)
        
        csv = pd.read_csv(path)
        csv_data = csv['DATA Sample']
        data = np.array(csv_data)
        signals = []
        count = 1
        peaks =  biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate = 200)[0]
        for i in (peaks[1:-1]):
           d1 = abs(peaks[count - 1] - i)
           d2 = abs(peaks[count + 1]- i)
           x = peaks[count - 1] + d1//2
           y = peaks[count + 1] - d2//2
           signal = data[x:y]
           signals.append(signal)
           count += 1
           indices.append((x,y))

            
        for count, i in enumerate(signals):
            fig = plt.figure(frameon=False)
            plt.plot(i) 
            plt.xticks([]), plt.yticks([])
            for spine in plt.gca().spines.values():
                spine.set_visible(False)

            save_file = 'fig' + '.png'
            fig.savefig(save_file)
            im_gray = cv2.imread(save_file, cv2.IMREAD_GRAYSCALE)
            im_gray = cv2.erode(im_gray,kernel,iterations = 1)
            im_gray = cv2.resize(im_gray, (128, 128), interpolation = cv2.INTER_LANCZOS4)
            cv2.imwrite(save_file, im_gray)
            im_gray = cv2.imread(save_file)
            pred = model.predict(im_gray.reshape((1, 128, 128, 3)))
            pred_class = pred.argmax(axis=-1)
            if pred_class == 0:
                APC.append(indices[count]) 
            elif pred_class == 1:
                NORMAL.append(indices[count])
        

       
        result = sorted(result.items(), key = lambda y: len(y[1]))[::-1]   
        output.append(result)
        data = {}
        data['filename'+ str(flag)] = str(path)
        data['result'+str(flag)] = str(result)

        json_filename = 'data.txt'
        with open(json_filename, 'a+') as outfile:
            json.dump(data, outfile) 
        flag+=1 
    
    with open(json_filename, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace('}{', ',')
    with open(json_filename, 'w') as file:
        file.write(filedata) 
    os.remove('fig.png')      
    return output
    
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        file_upload = []

        # Save the file to ./uploads
        print(file_upload)
        for f in request.files.getlist('file'):

            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
            print(file_path)
            if file_path[-4:] == '.csv':
                file_upload.append(file_path)
                f.save(file_path)
        print(file_upload)        
        # Prediction of Arrhythmia is done
        pred = Arrhythmia_Model(file_upload, model)
        result = str(pred)
        return result
    return None

if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
