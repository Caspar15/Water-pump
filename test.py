import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

def extract_features(data_path, sensor_column):
    # 讀取 CSV 文件
    data = pd.read_csv("/Users/chenbaiyu/Desktop/python/光點計畫/水泵/sensor.csv")

    # 使用指定的感測器列，並轉換為 NumPy 數組
    sensor_data = data[sensor_column].to_numpy()

    # 檢查並處理 NaN 值
    if np.isnan(sensor_data).any():
        print("NaN values found, filling with the mean of the column")
        sensor_data = np.nan_to_num(sensor_data, nan=np.nanmean(sensor_data))

    # 計算均方誤差 (MSE)
    mse = np.mean(np.square(sensor_data - np.mean(sensor_data)))

    # 計算標準差 (Standard Deviation)
    std_dev = np.std(sensor_data)

    # 峰值頻率 (Peak Frequency)
    fft = rfft(sensor_data)
    frequencies = rfftfreq(len(sensor_data), d=1/60)  # 假設數據每分鐘有60個樣本
    peak_frequency = frequencies[np.argmax(np.abs(fft))]

    return mse, peak_frequency, std_dev

def plot_features(mse, peak_frequency, std_dev):
    # 特徵名稱和值
    features = ['MSE', 'Peak Frequency', 'Standard Deviation']
    values = [mse, peak_frequency, std_dev]

    # 繪製特徵
    plt.figure(figsize=(10, 5))
    plt.bar(features, values, color=['blue', 'green', 'red'])
    plt.title('Features')
    plt.xlabel('Feature')
    plt.ylabel('Value')
    plt.show()

# 使用的資料
data_path = '/Users/chenbaiyu/Desktop/python/光點計畫/水泵/sensor.csv'  # 路徑請根據實際情況修改
sensor_column = 'sensor_02'  # 選擇要查看的感測器列
mse, peak_frequency, std_dev = extract_features(data_path, sensor_column)

# 繪製 MSE, Peak Frequency 和 Standard Deviation
plot_features(mse, peak_frequency, std_dev)
