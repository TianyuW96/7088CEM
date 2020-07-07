import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
import datetime
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

start_time = datetime.datetime.now()

# Load data
stock = pd.read_csv('./SH600276.csv')

# Select 300 data as the test set.
# Select the daily opening price in the data as the prediction data.
training_set = stock.iloc[0: 2427-300, 2: 3].values
test_set = stock.iloc[2427-300:, 2:3].values

# Normalize the data
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
test_set = sc.transform(test_set)

x_train = []
y_train = []

x_test = []
y_test = []

# Traverse the training set.
# Extract the opening price of 100 days as the input feature.
# And the opening price of the 101st day as the label.
for i in range(100, len(training_set_scaled)):
    x_train.append(training_set_scaled[i-100: i, 0])
    y_train.append(training_set_scaled[i, 0])

# Shuffle the training data
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

# Change the training set data type from list to array
x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], 100, 1))

# Traverse the test set.
# Extract the opening price of 100 days as the input feature.
# And the opening price of the 101st day as the label.
for i in range(100, len(test_set)):
    x_test.append(test_set[i-100: i, 0])
    y_test.append(test_set[i, 0])

# Change the test set data type from list to array
x_test = np.array(x_test)
y_test = np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 100, 1))

# Build a neural network
model = tf.keras.Sequential([
    LSTM(80, return_sequences=True),        # On the first loop calculation side, the memory is set to 80.
    Dropout(0.2),                           # To prevent overfitting, use a Dropout of 0.2
    LSTM(100),                              # On the second-level loop calculation side, the memory is set to 100.
    Dropout(0.2),
    Dense(1)                                # Output a forecast data
])

# Configure training methods
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='mean_squared_error',
    metrics=['accuracy']
)

checkpoint_save_path = "./checkpoint/LSTM.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('---------------------load the model----------------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_save_path,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss'
)

history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=100,
    validation_data=(x_test, y_test),
    validation_freq=1,
    callbacks=[cp_callback]
)

model.summary()

file = open('./LSTM-weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

# Test set input model for prediction
predicted_stock_price = model.predict(x_test)
# Restore forecast data
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# Restore really data
real_stock_price = sc.inverse_transform(test_set[100:])

# Calculate MSE
mse = mean_squared_error(predicted_stock_price, real_stock_price)
# Calculate RMSE
rmse = math.sqrt(mean_squared_error(predicted_stock_price,real_stock_price))
# Calculate MAE
mae = mean_absolute_error(predicted_stock_price, real_stock_price)

real_price = real_stock_price.tolist()
predicted_price = predicted_stock_price.tolist()

end_time = datetime.datetime.now()

print('-----------------------Error results-----------------------')
print('MSE: %.6f' % mse)
print('RMSE: %.6f' % rmse)
print('MAE: %.6f' % mae)

print('-------------------Operational efficiency-------------------')
print('Start time: ', start_time)
print('End time: ', end_time)
print('Running time: ', (end_time-start_time).seconds)

print('-----------------------Forecast results-----------------------')
print('Real stock price: \n', real_price[-6:-1])
print('Predicted stock price: \n', predicted_price[-6: -1])

loss = history.history['loss']
val_loss = history.history['val_loss']

# Loss function graph
plt.plot(loss, color='#FF8C00', linewidth=1.5, label='Training Loss')
plt.plot(val_loss, color='#00CDCD', linewidth=1.5, label='Validation Loss')
plt.title('SH600276 Training and Validation Loss')
plt.legend()
plt.show()

# Forecast curve
plt.plot(real_stock_price, color='#FF8C00', linewidth=1.5, label='Stock Price')
plt.plot(predicted_stock_price, color='#00CDCD', linewidth=1.5, label='Predicted Stock Price')
plt.title('SH600276 Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

