''' 
Dbcreator.py
Created by Shreyas Hukkeri
'''
import argparse
import csv
import dicom
import gzip
import numpy as np
import os
from os.path import isfile, join, isdir
import scipy.misc
import sys
import h5py

def read_in_one_image(path_img, name_img, matrix_size, data_aug=False):
    # Setting up the filepaths and opening up the format.
    #filepath_temp = join(path_img, 'temp.dcm')
    filepath_img = join(path_img, name_img)
    # Reading/uncompressing/writing
    #if isfile(filepath_temp):
    #    remove(filepath_temp)
    #with gzip.open(filepath_img, 'rb') as f_gzip:
    #    file_content = f_gzip.read()
    #    with open(filepath_temp, 'w') as f_dcm:
    #        f_dcm.write(file_content)
    # Reading in dicom file to ndarray and processing
    if os.path.exists(filepath_img+'.gz'):
        dicom_content = dicom.read_file(gzip.open(filepath_img+'.gz', 'rb'))
    else:
        dicom_content = dicom.read_file(filepath_img)
        
    img = dicom_content.pixel_array
    img = scipy.misc.imresize(img, (matrix_size, matrix_size), interp='cubic')
    img = img.astype(np.float32)
    img -= np.mean(img)
    img /= np.std(img)
    # Removing temporary file.
    #remove(filepath_temp)
    # Let's do some stochastic data augmentation.
    if not data_aug:
        return img
    if np.random.rand() > 0.5:                                #flip left-right
        img = np.fliplr(img)
    num_rot = np.random.choice(4)                             #rotate 90 randomly
    img = np.rot90(img, num_rot)
    up_bound = np.random.choice(174)                          #zero out square
    right_bound = np.random.choice(174)
    img[up_bound:(up_bound+50), right_bound:(right_bound+50)] = 0.0
    return img


if __name__ == '__main__':

	args = sys.argv
	parser = argparse.ArgumentParser(description = "Create the Database!")
	parser.add_argument("--pf", dest="path_data", type=str, default="/trainingData")
	parser.add_argument("--csv1", dest="csv1", type=str, default="/metadata/images_crosswalk.tsv")
	parser.add_argument("--csv2", dest="csv2", type=str, default="/metadata/exams_metadata.tsv")
	opts = parser.parse_args(args[1:])

	matrix_size = 200

	dict_img_to_patside = {}
	counter = 0
	with open(opts.csv1, 'rU') as file_crosswalk:
		reader_crosswalk = csv.reader(file_crosswalk, delimiter='\t')
		for row in reader_crosswalk:
			if counter == 0:
				counter += 1
				continue
			dict_img_to_patside[row[5].strip()] = (row[0].strip(), row[4].strip())

	dict_tuple_to_cancer = {}
	counter = 0
	with open(opts.csv2, 'rU') as file_metadata:
		reader_metadata = csv.reader(file_metadata, delimiter='\t')
		for row in reader_metadata:
			if counter == 0:
				counter += 1
				continue
			if row[3] == ".":
				dict_tuple_to_cancer[(row[0].strip(), 'R')] = int(row[4])
				continue
			elif row[4] == ".":
				dict_tuple_to_cancer[(row[0].strip(), 'L')] = int(row[3])
				continue
			else:
				dict_tuple_to_cancer[(row[0].strip(), 'L')] = int(row[3])
				dict_tuple_to_cancer[(row[0].strip(), 'R')] = int(row[4])
				continue
	X_tot = []
	Y_tot = []
	for img_name in dict_img_to_patside:
		X_tot.append(img_name)
		Y_tot.append(dict_tuple_to_cancer[dict_img_to_patside[img_name]])


	
	lenX = len(X_tot)

	X = np.zeros((lenX, matrix_size, matrix_size, 1), dtype=np.float32)
	Y = np.zeros((lenX, 2), dtype=np.int64)

	for num in range(lenX):
		X[num, :, :, 0] = read_in_one_image(opts.path_data, X_tot[num], matrix_size, data_aug=False)
		if Y_tot[num] == 0:
			Y[num] = [1,0]
		elif Y_tot[num] == 1:
			Y[num] = [0,1]
	print("Read all images into array.")

	h5f = h5py.File('data.h5', 'w')
	print("Started creating dataset!")
	print("Creating X datset...")
	h5f.create_dataset('X', data=X)
	print("Creating Y datset...")
	h5f.create_dataset('Y', data=Y)
	h5f.close()
	print("Dataset creation complete!")
