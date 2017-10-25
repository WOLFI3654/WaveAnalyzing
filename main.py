import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

train = true
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
    # Creating (num_of_data * batch_size, time_series, feature_num) data
    # num_of_data * batch_size = total number of data
    data = np.random.normal(0, 0.1, [num_of_data * batch_size, time_series, feature_num])
    label = np.random.randint(2, size=(num_of_data * batch_size))
    return data, label

# Change these code here to fit in your case.
print('Loading data...')
x_train, y_train = make_fake_data(num_of_data, batch_size, time_series, feature_num)
x_test, y_test = make_fake_data(num_of_data, batch_size, time_series, feature_num)
#
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
# Sequential is a nice keras neural network wrapper
model = Sequential()
# LSTM is a variant of RNN. LSTM performs generally better than RNN in all tasks.
model.add(LSTM(128, input_shape=(time_series, feature_num)))
# Linear layer which maps 128 -> 1, 1 is for binary classification.
model.add(Dense(1, activation='sigmoid'))

# Compile the whole model, binary_crossentropy for binary classification
# Adam is a generally nice optimizer
# metrics = ['accuracy'] to measure the result in accuracy
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

if train
    print('Train...')
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=5,
            )
    score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    model.save_weights("model")
else
    model.load_weights("model")
    # start using


