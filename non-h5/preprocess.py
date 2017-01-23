import os
import png
import dicom
import gzip
import csv
from multiprocessing import Pool
from os.path import join, basename
import scipy.misc
from scipy.misc import imsave

counter = 0
dict_img_to_patside = {}

matrix_size = 1000

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
print(len(X_tot))

def convert(f):
	dcm_file = str(X_tot[f])
	filepath_img = join('/trainingData', dcm_file)
	if os.path.exists(filepath_img+'.gz'):
		dicom_content = dicom.read_file(gzip.open(filepath_img+'.gz', 'rb'))
	else:
		dicom_content = dicom.read_file(filepath_img)
	img = dicom_content.pixel_array
	img = scipy.misc.imresize(img, (matrix_size, matrix_size), interp='cubic')

	imsave('/preprocessedData/' + os.path.splitext(dcm_file)[0] + '.jpg', img)


pool = Pool(processes=8)
inputs = range(len(X_tot))
print(inputs)
result = pool.map(convert, inputs)