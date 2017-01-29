from PIL import Image #, ImageChops
import csv
import os
from multiprocessing import Pool

def add_to_file(X, Y):
	with open('data.txt','a+') as g:
		g.write('pData/' + str(X) + '.jpg' + '\t' + str(Y) + '\n')
	g.close()

def preprocess(num):
	if data[num][2] == "NORM":
		add_to_file(data[num][0], 0) #Write to file as no tumor
	else:
		#Check whether Benign or Malignant
		if data[num][3] == "B":
			add_to_file(data[num][0], 1)#Write to file as Benign
		else:
			add_to_file(data[num][0], 2)#Write to file as Malignant

	Image.open(str('all-mias/' + data[num][0] + '.pgm')).save(str('pData/' + data[num][0] + '.jpg'))

f = open('mias.csv', 'rU')
data = list(csv.reader(f, dialect='excel'))

pool = Pool(processes=4)
inputs = range(len(data))
result = pool.map(preprocess, inputs)

f.close()