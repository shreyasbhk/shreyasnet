#Alexnet
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, image_preloader
import tflearn.datasets.mnist as mnist
import time

start_time = time.time()
# Data loading and preprocessing
X, Y = image_preloader('data.txt', image_shape=(900, 900),   mode='file', categorical_labels=True,   normalize=False)

# Building 'AlexNet'
network = input_data(shape=[None, 900, 900])
network = conv_1d(network, 96, 11, strides=4, activation='relu')
network = max_pool_1d(network, 1, strides=2)
#network = local_response_normalization(network)
network = conv_1d(network, 256, 5, activation='relu')
network = max_pool_1d(network, 1, strides=2)
#network = local_response_normalization(network)
network = conv_1d(network, 384, 3, activation='relu')
network = conv_1d(network, 384, 3, activation='relu')
network = conv_1d(network, 256, 3, activation='relu')
network = max_pool_1d(network, 1, strides=2)
#network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 3, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, tensorboard_verbose=3, tensorboard_dir='Tensorboard/')
model.fit(X, Y, n_epoch=5, validation_set=0.2, show_metric=True, batch_size=20, 
	shuffle=False, run_id='AlexNet')

model.save('Models/alexnet.tflearn')

end_time = time.time()
print("Training Time:")
print(end_time - start_time)

data = np.zeros((1, 900, 900), dtype=np.float32)
values = [1, 2, 5, 10, 20, 25, 9, 18]
for num in values:
	data[0, :, :] = X[num]
	print('Real Value: ' + str(Y[num]))
	print('Predicted Value: ' + str(model.predict(data)))
