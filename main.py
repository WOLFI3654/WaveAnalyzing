import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

num_of_data = 100
batch_size = 32
time_series = 8
feature_num = 100
'''
Make fake data for time series case.
num_of_data is the number of all data points.
batch_size is the number of batch. 
Batch is the size that number of data points in one training process, 
a resonable batch_size(32 ~ 128) would make training smoother.
time_series is the recurrent data points in one stream.
feature_num is the number of features in one data point.
'''
def make_fake_data(num_of_data, batch_size, time_series, feature_num):
    data = np.random.normal(0, 0.1, [num_of_data, batch_size, time_series, feature_num])
    label = np.random.randint(2, size=(num_of_data, batch_size))
    return data, label

print('Loading data...')
x_train, y_train = make_fake_data(num_of_data, batch_size, time_series, feature_num)
x_test, y_test = make_fake_data(num_of_data, batch_size, time_series, feature_num)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(time_series, feature_num)))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
