from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_cors import CORS
import os
from os.path import join, dirname, realpath
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__)

CORS(app)

#load the model
occ_model = tf.keras.models.load_model('occ_model.h5')
adr_model = tf.keras.models.load_model('adr_model.h5')
ari_model = tf.keras.models.load_model('ari_model.h5')
ori_model = tf.keras.models.load_model('ori_model.h5')
revpar_model = tf.keras.models.load_model('revpar_model.h5')
rgi_model = tf.keras.models.load_model('rgi_model.h5')

# enable debugging mode
app.config["DEBUG"] = True

# Upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER


# Root URL
@app.route('/')
def index():
     # Set The upload HTML template '\templates\index.html'
    return render_template('index.html')


# Get the uploaded files
@app.route("/predict/ocupancyrate", methods=['POST'])
def ocupancyrate():
    # get the uploaded file
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        # set the file path
        uploaded_file.save(file_path)

        y_pred_future = parseCSV(file_path, occ_model)
         
    return y_pred_future #redirect(url_for('index'))

@app.route("/predict/adr", methods=['POST'])
def adr():
    # get the uploaded file
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        # set the file path
        uploaded_file.save(file_path)

        y_pred_future = parseCSV(file_path, adr_model)
         
    return y_pred_future #redirect(url_for('index'))


@app.route("/predict/ari", methods=['POST'])
def ari():
    # get the uploaded file
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        # set the file path
        uploaded_file.save(file_path)

        y_pred_future = parseCSV(file_path, ari_model)
         
    return y_pred_future #redirect(url_for('index'))

@app.route("/predict/ori", methods=['POST'])
def ori():
    # get the uploaded file
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        # set the file path
        uploaded_file.save(file_path)

        y_pred_future = parseCSV(file_path, ori_model)
         
    return y_pred_future #redirect(url_for('index'))


@app.route("/predict/revpar", methods=['POST'])
def revpar():
    # get the uploaded file
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        # set the file path
        uploaded_file.save(file_path)

        y_pred_future = parseCSV(file_path, revpar_model)
         
    return y_pred_future #redirect(url_for('index'))


@app.route("/predict/rgi", methods=['POST'])
def rgi():
    # get the uploaded file
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        # set the file path
        uploaded_file.save(file_path)

        y_pred_future = parseCSV(file_path, rgi_model)
         
    return y_pred_future #redirect(url_for('index'))



def prepare_test_data(df, n_future, n_past):
    testX = []
    testY = []
    df = df.drop(['id'], axis=1)
    numerical_cols = list(df)[1:9]
    categorical_cols = list(df)[9:12]
    for i in categorical_cols:
        del df[i]
    df[numerical_cols] = df[numerical_cols].fillna(df.mean())
    test_dates = pd.to_datetime(df['dates'])
    df = df.drop(['dates'], axis=1)
    scaler = MinMaxScaler()
    df_for_testing_scaled = scaler.fit_transform(df)
    for i in range(n_past, len(df_for_testing_scaled) - n_future +1):
        testX.append(df_for_testing_scaled[i - n_past:i, 0:df_for_testing_scaled.shape[1]])
        testY.append(df_for_testing_scaled[i + n_future - 1:i + n_future, 0])
    
    testX, testY = np.array(testX), np.array(testY)

    return test_dates, df_for_testing_scaled, testX, testY



def parseCSV(filePath, model):
    df_test = pd.read_csv(filePath)
    n_future = 15
    n_past = 30
    results = []
    test_dates, df_for_testing_scaled, testX, testY = prepare_test_data(df_test, n_future, n_past)
    prediction = model_predict(model, testX)
    scaler = MinMaxScaler()
    df_new_scaled = scaler.fit_transform(df_for_testing_scaled)
    prediction_copies = np.repeat(prediction, df_new_scaled.shape[1], axis=-1)
    y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]
    
    print(y_pred_future)
    
    print(str(y_pred_future))

    for i in y_pred_future:
        results.append(i)
    
    return str(results)

def model_predict(model, testX):
    prediction = model.predict(testX)
    return prediction





if (__name__ == "__main__"):
     app.run(host='0.0.0.0', port = 8080, debug=False)
