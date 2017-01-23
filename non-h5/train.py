#ConCaDNet v4.0.0

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
The following is the code for building ConCaDNet, a Convolutional Neural 
Network (CNN) based on the inception model and built using the TFLearn Library. 

The code takes input in the form of matrix_sizexmatrix_state matrices from 
    a .h5 file. ConCaDNet is then trained on the data and the model is 
    saved as a .tflearn file.
'''

import sys
import tflearn
from tflearn.layers.conv import conv_1d, max_pool_1d #conv_2d, max_pool_2d
from tflearn.layers.core import input_data, fully_connected #dropout, 
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge
from tflearn.data_utils import to_categorical
from tflearn.data_utils import image_preloader
import time
import h5py


if __name__ == '__main__':

    start_time = time.time()
    matrix_size = 1000
    
    X, Y = image_preloader('/scratch/data.txt', image_shape=(matrix_size, matrix_size, 1),   mode='file', categorical_labels=True,   normalize=False)
    

    conv_input = input_data(shape=[None, matrix_size, matrix_size], name='input')
    
    conv = conv_1d(conv_input, 100, filter_size=50, activation='leaky_relu', strides=2)
    conv1 = conv_1d(conv_input, 50, 1, activation='leaky_relu', strides=1)
    conv1 = max_pool_1d(conv1, kernel_size=2, strides=2)
    
    convnet = merge([conv, conv1], mode='concat', axis=2)
    convnet = conv_1d(convnet, 30, filter_size=1, activation='relu')
    #convnet = dropout(convnet, 0.35) -- Currently disabled (can be included if generalization is necessary)

    convnet = fully_connected(convnet, 3, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=0.06, loss='categorical_crossentropy')
    
    model = tflearn.DNN(convnet, tensorboard_verbose=3, tensorboard_dir='Tensordboard/')
    model.fit(X, Y, n_epoch=5, validation_set=0.2, show_metric=True, batch_size=5, snapshot_step=100, 
        snapshot_epoch=False, run_id='Digital Mammography ConCaDNet')
    model.save('Models/model.tflearn')
    
    end_time = time.time()
    print("Training Time:")
    print(end_time - start_time)


    '''
    data = np.zeros((1, matrix_size, matrix_size, 1), dtype=np.float32)
    data[0, :, :, :] = X[15]
    print(Y[15])
    print(model.predict(data))
    
    '''