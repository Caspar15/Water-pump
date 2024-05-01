from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

app = Flask(__name__)

# 載入模型和數據
model_path = "/Users/chenbaiyu/Desktop/python/model.keras"
model = load_model(model_path)
data_path = "/Users/chenbaiyu/Desktop/python/光點計畫/水泵/sensor.csv"
data = pd.read_csv(data_path)

def preprocess_data(data):
    # 資料預處理
    data = data.drop(['Unnamed: 0', 'timestamp', 'sensor_00', 'sensor_15', 'sensor_50', 'sensor_51'], axis=1)
    data.fillna(method='ffill', inplace=True)
    
    conditions = [
        (data['machine_status'] == 'NORMAL'),
        (data['machine_status'] == 'BROKEN'),
        (data['machine_status'] == 'RECOVERING')
    ]
    choices = [1, 0, 0.5]
    data['Operation'] = np.select(conditions, choices, default=0)

    selected_sensors = ['sensor_04', 'sensor_06', 'sensor_07', 'sensor_08', 'sensor_09']
    df0 = data[selected_sensors + ['Operation']].copy()  # 使用 .copy() 來確保我們有一個數據的副本

    scaler_features = MinMaxScaler()
    df0.loc[:, selected_sensors] = scaler_features.fit_transform(df0[selected_sensors])  # 使用 .loc[] 進行賦值

    scaler_operation = MinMaxScaler()
    df0.loc[:, 'Operation'] = scaler_operation.fit_transform(df0[['Operation']])  # 使用 .loc[] 進行賦值

    return df0, scaler_features, scaler_operation

    # 資料預處理
    data = data.drop(['Unnamed: 0', 'timestamp', 'sensor_00', 'sensor_15', 'sensor_50', 'sensor_51'], axis=1)
    data.fillna(method='ffill', inplace=True)
    
    conditions = [
        (data['machine_status'] == 'NORMAL'),
        (data['machine_status'] == 'BROKEN'),
        (data['machine_status'] == 'RECOVERING')
    ]
    choices = [1, 0, 0.5]
    data['Operation'] = np.select(conditions, choices, default=0)

    selected_sensors = ['sensor_04', 'sensor_06', 'sensor_07', 'sensor_08', 'sensor_09']
    df0 = data[selected_sensors + ['Operation']]

    scaler_features = MinMaxScaler()
    df0[selected_sensors] = scaler_features.fit_transform(df0[selected_sensors])

    scaler_operation = MinMaxScaler()
    df0['Operation'] = scaler_operation.fit_transform(df0[['Operation']])

    return df0, scaler_features, scaler_operation

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    try:
        df0, _, scaler_operation = preprocess_data(data)
        test_x = df0.values.reshape((df0.shape[0], 1, df0.shape[1]))
        yhat = model.predict(test_x)
        inv_yhat = scaler_operation.inverse_transform(yhat)
        return jsonify({'actual': df0['Operation'].tolist(), 'predicted': inv_yhat.flatten().tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
