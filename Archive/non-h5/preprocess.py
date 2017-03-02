import os
import sys
import argparse
import png
import dicom
import gzip
import time
import csv
from math import isclose
from multiprocessing import Pool
from os.path import join, basename
import scipy.misc
from scipy.misc import imsave

start_time = time.time()
args = sys.argv
parser = argparse.ArgumentParser(description = "Do deep learning!")
parser.add_argument("--num_cpu", dest="num_cpu", type=str, default=12)
parser.add_argument("--ms", dest="matrix_size", type=int, default=1000)
opts = parser.parse_args(args[1:])


counter = 0
matrix_size = opts.matrix_size
dict_img_to_patside = {}
with open('/metadata/images_crosswalk.tsv', 'rU') as file_crosswalk:
		reader_crosswalk = csv.reader(file_crosswalk, delimiter='\t')
		for row in reader_crosswalk:
			if counter == 0:
				counter += 1
				continue
			dict_img_to_patside[row[5].strip()] = row[0].strip()
X_tot = []
for img_name in dict_img_to_patside:
	X_tot.append(img_name)
lenX = len(X_tot)
print("Number of Images: " + str(lenX))
print("Number of CPU Cores: " + str(opts.num_cpu))

num_img = 0

def convert(f):
	global num_img
	dcm_file = str(X_tot[f])
	filepath_img = join('/trainingData', dcm_file)
	if os.path.exists(filepath_img+'.gz'):
		dicom_content = dicom.read_file(gzip.open(filepath_img+'.gz', 'rb'))
	else:
		dicom_content = dicom.read_file(filepath_img)
	img = dicom_content.pixel_array
	img = scipy.misc.imresize(img, (matrix_size, matrix_size), interp='cubic')
	imsave('/preprocessedData/' + os.path.splitext(dcm_file)[0] + '.jpg', img)

	now_time = time.time()
	intervals = [5, 10, 25, 50, 100, 125, 150, 150, 300, 600, 1800, 3600, 5400, 7200, 10800, 14400, 18000, 21600, 25200, 28800]
	for i in intervals:
		if isclose((now_time - start_time), i, abs_tol=0.3):
			print("It has been " + str((now_time - start_time)) + " seconds")
			print("Number of Images processed: " + str(num_img))
			print("Number of Images Remaining: " + str((lenX - num_img)))
		else:
			continue
	num_img = num_img + int(opts.num_cpu)
	'''
	intervals = [0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 0.95]
	for i in intervals:
		if isclose(num_img, (lenX*i), abs_tol=0.5):
			now_time = time.time()
			print(str(i) + " '%' of the Data has been preprocessed after " + str((now_time - start_time)) + " seconds")
	'''

pool = Pool(processes=int(opts.num_cpu))
inputs = range(lenX)
print(inputs)
result = pool.map(convert, inputs)
