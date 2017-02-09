from PIL import Image, ImageChops
import csv
import os
import numpy as np
import scipy.misc
from multiprocessing import Pool

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((1020,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -25)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def reshape(im):
	return scipy.misc.imresize(im, (900, 900), interp='cubic')

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

	Image.fromarray(reshape(trim(Image.open(str('all-mias/' + data[num][0] + '.pgm'))))).save(str('pData/' + data[num][0] + '.jpg'))


f = open('mias.csv', 'rU')
data = list(csv.reader(f, dialect='excel'))

pool = Pool(processes=4)
inputs = range(len(data))
result = pool.map(preprocess, inputs)

f.close()