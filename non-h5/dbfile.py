#ConCaDNet v4.0.0
''' 
Dbcreator.py
Created by Shreyas Hukkeri
'''
import argparse
import csv
import dicom
import gzip
import os
from os.path import isfile, join, isdir, basename
import sys
import time
from multiprocessing import Pool

start_time = time.time()
args = sys.argv
parser = argparse.ArgumentParser(description = "Create the Database!")
parser.add_argument("--csv1", dest="csv1", type=str, default="/metadata/images_crosswalk.tsv")
parser.add_argument("--csv2", dest="csv2", type=str, default="/metadata/exams_metadata.tsv")
parser.add_argument("--num_cpu", dest="num_cpu", type=str, default=8)
opts = parser.parse_args(args[1:])

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
print('lenX:')
print(lenX)
print('lenY:')
print(lenY)

def dbc(num):
	with open('/scratch/data.txt','a+') as f:
		f.write('/preprocessedData/' + str(os.path.splitext(X_tot[num])[0]) + '.jpg' + '\t' + str(Y_tot[num]) + '\n')
	f.close()

pool = Pool(processes=opts.num_cpu)
inputs = range(lenX)
result = pool.map(dbc, inputs)

print("Dataset creation complete!")

end_time = time.time()
print("Database Creation Time:")
print(end_time - start_time)


