#ConCaDNet v6.0.0

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

import argparse
import sys
import tflearn
import numpy as np
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge
from tflearn.data_utils import to_categorical, image_preloader
import time
import h5py

start_time = time.time()
X, Y = image_preloader('data.txt', image_shape=(1024, 1024),   mode='file', categorical_labels=True,   normalize=False)

conv_input = input_data(shape=[None, 1024, 1024], name='input')
conv = conv_1d(conv_input, 10, filter_size=50, activation='leaky_relu', strides=1)
conv1 = conv_1d(conv_input, 5, filter_size=1, activation='leaky_relu', strides=1)
conv1 = max_pool_1d(conv1, kernel_size=2, strides=1)
convnet = merge([conv, conv1], mode='concat', axis=2)
convnet = conv_1d(convnet, 30, filter_size=1, activation='leaky_relu')
convnet = dropout(convnet, 0.5)
convnet = fully_connected(convnet, 3, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')
model = tflearn.DNN(convnet, tensorboard_verbose=3, tensorboard_dir='Tensordboard/')
model.fit(X, Y, n_epoch=5, validation_set=0.2, show_metric=True, batch_size=20, shuffle=True,
	snapshot_epoch=True, run_id='Digital Mammography ConCaDNet')

model.save('Model/model.tflearn')

end_time = time.time()
print("Training Time:")
print(end_time - start_time)

data = np.zeros((1, 1024, 1024), dtype=np.float32)
data[0, :, :] = X[1]
print('Real Value: ' + str(Y[1]))
print('Predicted Value: ' + str(model.predict(data)))
