#ConCaDNet v4.0.0
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
from os.path import isfile, join, isdir, basename
import scipy.misc
from sklearn.model_selection import train_test_split
import sys
import h5py
import time
from multiprocessing import Pool

if __name__ == '__main__':

	start_time = time.time()
	args = sys.argv
	parser = argparse.ArgumentParser(description = "Create the Database!")
	parser.add_argument("--csv1", dest="csv1", type=str, default="/metadata/images_crosswalk.tsv")
	parser.add_argument("--csv2", dest="csv2", type=str, default="/metadata/exams_metadata.tsv")
	opts = parser.parse_args(args[1:])

	matrix_size = 500
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
			if counter ==0:
				counter += 1
				continue
			else:	
				if row[4] == '0':
					dict_tuple_to_cancer[(row[0].strip(), 'R')] = 0
				else:
					if row[6] == '1':
						dict_tuple_to_cancer[(row[0].strip(), 'R')] = 2
					elif row[6] == '0':
						dict_tuple_to_cancer[(row[0].strip(), 'R')] = 1

				if row[3] == '0':
					dict_tuple_to_cancer[(row[0].strip(), 'L')] = 0
				else:
					if row[5] == '1':
						dict_tuple_to_cancer[(row[0].strip(), 'L')] = 2
					elif row[5] == '0':
						dict_tuple_to_cancer[(row[0].strip(), 'L')] = 1
				
				if row[3] == "." and row[4] == ".":
					continue
				elif row[3] == ".":
					if row[4] == '0':
						dict_tuple_to_cancer[(row[0].strip(), 'R')] = 0
					else:
						if row[6] == '1':
							dict_tuple_to_cancer[(row[0].strip(), 'R')] = 2
						else:
							dict_tuple_to_cancer[(row[0].strip(), 'R')] = 1
					continue

				elif row[4] == ".":
					if row[3] == 0:
						dict_tuple_to_cancer[(row[0].strip(), 'L')] = 0
					else:
						if row[5] == '1':
							dict_tuple_to_cancer[(row[0].strip(), 'L')] = 2
						else:
							dict_tuple_to_cancer[(row[0].strip(), 'L')] = 1
					continue

	X_tot = []
	Y_tot = []
	for img_name in dict_img_to_patside:
		X_tot.append(img_name)
		Y_tot.append(dict_tuple_to_cancer[dict_img_to_patside[img_name]])
	
	lenX = len(X_tot)
	lenY = len(Y_tot)
	print(lenX)
	print(lenY)

	
	with open('/scratch/data.txt','a+') as f:
		for num in range(lenX):
			f.write('/preprocessedData/' + str(basename(X_tot[num])) + '.jpg' + '\t' + str(Y_tot[num]) + '\n')

	f.close
	'''
	pool = Pool(processes=8)
	inputs = range(lenX)
	result = pool.map(dbc, inputs)
	'''

	print("Dataset creation complete!")

	end_time = time.time()
	print("Database Creation Time:")
	print(end_time - start_time)


