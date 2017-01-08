#ShreyasNET v2.1.8

#Copyright (c) 2016 Shreyas Hukkeri
#
#Permission is hereby granted, free of charge, to any person obtaining 
#a copy of this software and associated documentation files (the "Software"), 
#to deal in the Software without restriction, including without limitation 
#the rights to use, copy, modify, merge, publish, distribute, sublicense, 
#and/or sell copies of the Software, and to permit persons to whom the Software 
#is furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all 
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
#EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
#IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
#DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
#ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
#OTHER DEALINGS IN THE SOFTWARE.

'''
The following is the code for building ShreyasNET, a Convolutional Neural 
Network (CNN) based on the inception model and built using the TFLearn Library. 

The code takes input in the form of matrix_sizexmatrix_state matrices from 
    a .h5 file. ShreyasNET is then trained on the data and the model is 
    saved as a .tflearn file.
'''

import sys
from os import remove
from os.path import isfile
import h5py
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, flatten, reshape
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge
import time
import numpy as np
from tflearn.datasets import cifar10
from tflearn.data_utils import shuffle, to_categorical

if __name__ == '__main__':

    matrix_size = 32
    (X, Y), (X_test, Y_test) = cifar10.load_data()
    X, Y = shuffle(X, Y)
    Y = to_categorical(Y, 10)
    Y_test = to_categorical(Y_test, 10)

    start_time = time.time()
    

    conv_input = input_data(shape=[None, matrix_size,matrix_size,3], name='input')
    
    conv = conv_2d(conv_input, 32, 3, activation='leaky_relu')
    conv = max_pool_2d(conv, 4)
    conv1 = conv_2d(conv_input, 64, 3, activation='leaky_relu')
    conv1 = max_pool_2d(conv1, 2)
    conv2 = conv_2d(conv1, 128, 3, activation='leaky_relu')
    conv2 = max_pool_2d(conv2, 2)
    conv3 = max_pool_2d(conv1, 2)
    conv1 = max_pool_2d(conv1, 2)
    print(conv)
    print(conv1)
    print(conv2)
    print(conv3)
    
    convnet = merge([conv, conv1, conv2, conv3], mode='concat', axis=3)
    print(convnet)
    #convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, 10, activation='softmax')
    #convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')
    
    model = tflearn.DNN(convnet, tensorboard_verbose=3, tensorboard_dir='Tensordboard/')
    model.fit(X, Y, n_epoch=10, validation_set=0.2, show_metric=True, batch_size=500, snapshot_step=500, 
        snapshot_epoch=False, run_id='shreyasnet_v2.1.8_run-3')
    model.save('Models/model_v2.1.8_run-3.tflearn')
    
    end_time = time.time()
    print("Time:")
    print(end_time - start_time)


    '''
    data = np.zeros((1, matrix_size, matrix_size, 1), dtype=np.float32)
    data[0, :, :, :] = X[15]
    print(Y[15])
    print(model.predict(data))
    
    '''