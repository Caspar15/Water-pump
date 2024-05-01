import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Input
from keras.callbacks import ModelCheckpoint

# CSV 檔案位置
data_path = "/Users/chenbaiyu/Desktop/python/光點計畫/水泵/sensor.csv"

# 讀取 CSV 文件
data = pd.read_csv(data_path)

# 從 DATA 刪除不需要的列
data.drop(['Unnamed: 0', 'timestamp', 'sensor_00', 'sensor_15', 'sensor_50', 'sensor_51'], axis=1, inplace=True)

# 缺失值處理：前向填充
data.fillna(method='ffill', inplace=True)

# 建立 Operation 列
conditions = [
     (data['machine_status'] == 'NORMAL'),
     (data['machine_status'] == 'BROKEN'),
     (data['machine_status'] == 'RECOVERING')
]
choices = [1, 0, 0.5]
data['Operation'] = np.select(conditions, choices, default=0)

# 資料視覺化
data.plot(subplots=True, sharex=True, figsize=(20, 50))
plt.show()

# 特徵選擇
selected_sensors = ['sensor_04', 'sensor_06', 'sensor_07', 'sensor_08', 'sensor_09']
df0 = pd.DataFrame(data, columns=['Operation'] + selected_sensors)

# 數據標準化
scaler_features = MinMaxScaler()
scaled_features = scaler_features.fit_transform(df0[selected_sensors])
df0.loc[:, selected_sensors] = scaled_features

# Operation 欄位的標準化
scaler_operation = MinMaxScaler()
df0['Operation'] = scaler_operation.fit_transform(df0[['Operation']])

# 轉換為監督式學習資料集
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
     n_vars = data.shape[1]
     df = pd.DataFrame(data)
     cols, names = [], []
     # 輸入序列 (t-n, ... t-1)
     for i in range(n_in, 0, -1):
         cols.append(df.shift(i))
         names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
     # 預測序列 (t, t+1, ... t+n)
     for i in range(0, n_out):
         cols.append(df.shift(-i))
         if i == 0:
             names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
         else:
             names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
     # 合併所有列
     agg = pd.concat(cols, axis=1)
     agg.columns = names
     # 刪除含有 NaN 值的行
     if dropnan:
         agg.dropna(inplace=True)
     return agg

reframed = series_to_supervised(df0.values, 1, 1)
reframed.drop(reframed.columns[list(range(df0.shape[1]+1, 2*df0.shape[1]))], axis=1, inplace=True)

# 分割資料為訓練集和測試集
values = reframed.values
n_train_time = int(len(values) * 0.8)
train = values[:n_train_time, :]
test = values[n_train_time:, :]

# 分割為輸入和輸出
train_x, train_y = train[:, :-1], train[:, -1]
test_x, test_y = test[:, :-1], test[:, -1]

# 重塑輸入為 [樣本, 時間步, 特徵]
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

# 建構 LSTM 模型
model = Sequential()
model.add(Input(shape=(train_x.shape[1], train_x.shape[2])))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 使用模型檢查點保存訓練最佳模型
checkpoint = ModelCheckpoint('model.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# 訓練模型
history = model.fit(train_x, train_y, epochs=150, batch_size=70, validation_data=(test_x, test_y), verbose=2, shuffle=False, callbacks=[checkpoint])

# 繪製訓練歷史
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# 載入已儲存的模型
from keras.models import load_model
model = load_model('model.keras')

# 進行預測
yhat = model.predict(test_x)

# 僅對預測的 Operation 欄位進行逆標準化
inv_yhat = scaler_operation.inverse_transform(yhat)

# 逆轉 test_y 以計算 RMSE
test_y = test_y.reshape((len(test_y), 1))
inv_y = scaler_operation.inverse_transform(test_y)

# 計算 RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)