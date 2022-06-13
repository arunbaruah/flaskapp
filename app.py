from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from os.path import join, dirname, realpath
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__)

#load the model
model = tf.keras.models.load_model('occ_model.h5')


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
@app.route("/predict", methods=['POST'])
def uploadFiles():
    # get the uploaded file
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        # set the file path
        uploaded_file.save(file_path)
        y_pred_future = parseCSV(file_path)
         
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



def parseCSV(filePath):
      # Use Pandas to parse the CSV file
    df_test = pd.read_csv(filePath)
    n_future = 15
    n_past = 30
    test_dates, df_for_testing_scaled, testX, testY = prepare_test_data(df_test, n_future, n_past)
    prediction = model.predict(testX)
    scaler = MinMaxScaler()
    df_new_scaled = scaler.fit_transform(df_for_testing_scaled)
    prediction_copies = np.repeat(prediction, df_new_scaled.shape[1], axis=-1)
    y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]
    return str(y_pred_future)

      




if (__name__ == "__main__"):
     app.run(port = 8080)