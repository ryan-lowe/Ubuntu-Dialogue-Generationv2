from __future__ import division
import math
from random import randint
from random import shuffle
import os
import csv
import sys

segments = 1500

def chunkIt(seq, num):
	avg = len(seq) / float(num)
	out = []
	last = 0.0
	while last < len(seq):
		out.append(seq[int(last):int(last + avg)])
		last += avg
	return out

def writeFiles(filename, data):
	with open(filename,'w') as out:
  		csv_out = csv.writer(out)
  		for row in data:
			csv_out.writerow(row)

def getFilelist(path):
	filelist = []
	folders = [f for f in os.listdir(path)]
	for folder in folders:
		filepath = path + '/' + folder
		files = [f for f in os.listdir(filepath)]
		for infile in files:
			filelist.append((folder, infile))
	shuffle(filelist)
	return filelist

def runParFiles(segments):
	filelist = getFilelist('/gs/project/jim-594-aa/rlowe/dialogs50/')
	newtest = chunkIt(filelist,segments)

	for i in range(len(newtest)):
		filename = './dialogsegs/dialoguesegment_' + str(i) + '.csv'
		writeFiles(filename,newtest[i])

if __name__ == '__main__':
	runParFiles(segments)
