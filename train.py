import sys
from os import remove
from os.path import isfile
import h5py
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge
import time
import numpy as np


if __name__ == '__main__':

    matrix_size = 200
    h5f = h5py.File('data.h5', 'r')
    X = h5f['X']
    Y = h5f['Y']

    start_time = time.time()
    

    conv_input = input_data(shape=[None, matrix_size,matrix_size,1], name='input')
    
    conv = conv_2d(conv_input, 1, 50, activation='relu', strides=5)
    conv1 = conv_2d(conv_input, 1, 1, activation='relu', strides=1)
    conv = flatten(conv)
    conv1 = flatten(conv1)
    
    convnet = merge([conv, conv1], mode='concat', axis=1)
    convnet = dropout(convnet, 0.35)

    convnet = fully_connected(convnet, 10, activation='softmax')
    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=0.006, loss='categorical_crossentropy')
    
    model = tflearn.DNN(convnet, tensorboard_verbose=3)
    #model.fit(X, Y, n_epoch=2, validation_set=0.2, show_metric=True, batch_size=20, snapshot_step=4, snapshot_epoch=False, run_id='shreyasnet_v6.2-Trial_1')

    model.load('model.tflearn')
    data = np.zeros((1, matrix_size, matrix_size, 1), dtype=np.float32)
    data[0, :, :, :] = X[15]
    print(Y[15])
    print(model.predict(data))
    end_time = time.time()
    print("Time:")
    print(end_time - start_time)


    h5f.close()