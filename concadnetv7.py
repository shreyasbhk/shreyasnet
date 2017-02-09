#ConCaDNet v7.0.0

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
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge
from tflearn.data_utils import to_categorical, image_preloader
from tflearn.data_augmentation import ImageAugmentation
import time
import h5py

start_time = time.time()

X_dim = 1024
Y_dim = 512
'''
h5f = h5py.File('data.h5', 'r')
X = h5f['X']
Y = to_categorical(h5f['Y'], 2)
'''

testX, testY = image_preloader('test.txt', image_shape=(Y_dim, X_dim), mode='file', categorical_labels=True, normalize=False)

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()

conv_input = input_data(shape=[None, X_dim, Y_dim, 1], name='input', data_augmentation=img_aug)

conv_1 = conv_2d(conv_input, 16, filter_size=64, activation='leaky_relu', strides=2) #Outputs 512x256x16
conv_2 = conv_2d(conv_input, 32, filter_size=32, activation='leaky_relu', strides=1) #Outputs 1024x512x32
max_1 = max_pool_2d(conv_2, kernel_size=3, strides=2) #Outputs 512x256x32
conv_3 = conv_2d(conv_input, 16, filter_size=5, activation='leaky_relu', strides=1) #Outputs 1024x512x16
max_2 = max_pool_2d(conv_3, kernel_size=2, strides=2) #Outputs 512x256x16
concat = merge([conv_1, max_1, max_2], mode='concat', axis=3) #Ouputs 512x256x64

### 3x3 Convolution Inception Module
inception_1 = conv_2d(concat, 32, filter_size=1, activation='leaky_relu', strides=1) #Outputs 512x256x32
inception_1 = conv_2d(inception_1, 32, filter_size=2, activation='leaky_relu', strides=1) #Outputs 512x256x32
inception_1 = max_pool_2d(inception_1, kernel_size=2, strides=2) #Outputs 256x128x32
inception_1 = conv_2d(inception_1, 64, filter_size=1, activation='leaky_relu', strides=1) #Outputs 256x128x64

max_3 = max_pool_2d(inception_1, kernel_size=3, strides=2) #Outputs 128x64x64

### 5x5 Convolution Inception Module 1
inception_2 = conv_2d(max_3, 32, filter_size=1, activation='leaky_relu', strides=1) #Outputs 128x64x32
inception_2 = conv_2d(inception_2, 32, filter_size=5, activation='leaky_relu', strides=1) #Outputs 128x64x32
inception_2 = max_pool_2d(inception_2, kernel_size=2, strides=2) #Outputs 64x32x32
inception_2 = conv_2d(inception_2, 64, filter_size=1, activation='leaky_relu', strides=1) #Outputs 64x32x64

max_4 = max_pool_2d(inception_2, kernel_size=3, strides=2) #Outputs 32x16x64

### 5x5 Convolution Inception Module 2
inception_3 = conv_2d(max_4, 32, filter_size=1, activation='leaky_relu', strides=1) #Outputs 32x16x32
inception_3 = conv_2d(inception_3, 32, filter_size=5, activation='leaky_relu', strides=1) #Outputs 32x16x32
inception_3 = max_pool_2d(inception_3, kernel_size=2, strides=2) #Outputs 16x8x32
inception_3 = conv_2d(inception_3, 64, filter_size=1, activation='leaky_relu', strides=1) #Outputs 16x8x64

max_5 = max_pool_2d(inception_3, kernel_size=5, strides=4) #Outputs 4x2x64

drop = dropout(max_5, 0.5)

convnet = fully_connected(drop, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')
model = tflearn.DNN(convnet, tensorboard_verbose=3, tensorboard_dir='Tensordboard/')
#model.fit(X, Y, n_epoch=10, validation_set=0.2, show_metric=True, batch_size=5, shuffle=True,
#	snapshot_epoch=True, run_id='ConCaDNetV7_Run1')

model.load('Models/feb3_1.tflearn')

end_time = time.time()
print("Training Time:")
print(end_time - start_time)
'''
#data = np.zeros((None, X_dim, Y_dim, 1), dtype=np.float32)
values = [1, 2, 5, 10, 20, 25, 9, 18, 21, 22 ,23 ,24, 31]
for num in values:
	#data[0, :, :, 0] = X[num]
	print('Real Value: ' + str(Y[num]))
	print('Predicted Value: ' + str(model.predict([X[num]])))
'''
data = np.zeros((len(testX), X_dim, Y_dim, 1), dtype=np.float32)
#Model Evaluation
for i in range(len(testX)):
	data[i, :, :, 0] = testX[i]

print("Model Evaluating:")
print(model.evaluate(data, testY, batch_size=5))

