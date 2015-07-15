from __future__ import division
import math
import time
import os
import csv
import sys
import re
from twokenize import tokenize
import nltk
from sklearn.externals import joblib



def is_number(s):
  try:
    float(s)
    return True
  except ValueError:
    return False

def is_url(s):
    return s.startswith('http://') or s.startswith('https://') or s.startswith('ftp://') \
            or s.startswith('ftps://') or s.startswith('smb://')

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"\'m", " \'m", string) 
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r"`", " ` ", string)
    string = re.sub(r",", " , ", string) 
    return string.strip() 

def process_token(c, word):
    """
    Use NLTK to replace named entities with generic tags.
    Also replace URLs, numbers, and paths.
    """
    nodelist = ['PERSON', 'ORGANIZATION', 'GPE', 'LOCATION', 'FACILITY', 'GSP']
    if hasattr(c, 'label'):
        if c.label() in nodelist:
            return "__%s__" % c.label()
    if is_url(word):
        return "__URL__"
    elif is_number(word):
        return "__NUMBER__"
    elif os.path.isabs(word):
        return "__PATH__"
    return word

def process_line(s, clean_string=True):
    """
    Processes a line by iteratively calling process_token.
    """
    if clean_string:
        s = clean_str(s)
    tokens = tokenize(s)
    sent = nltk.pos_tag(tokens)
    chunks = nltk.ne_chunk(sent, binary=False)
    return [process_token(c,token).lower().encode('UTF-8') for c,token in map(None, chunks, tokens)]

def writeFiles(csvname, data, listbool=False, overwrite=False):
	"""
	Writes to .csv files (overwrite optional).
	"""
	with open(csvname,'a+') as out:
		csv_out = csv.writer(out)
		for row in data:
		  if listbool:
		    for col in row:
		      csv_out.writerow(col)
		  else:
		    csv_out.writerow(row)		

def getFilesFromCsv(files):
	"""
	Produces a list of files from some csv file
	"""
	filelist = []
	with open(files, 'r') as c1:
		c1 = csv.reader(c1, delimiter = ',')
		for f, folder in c1:
			filelist.append([f, folder])  
	return filelist

def makeBadfileDict(badfiles):
	"""
	Produces a dictionary of badfiles
	"""
	filedict = {}
	with open(badfiles, 'r') as c1:
		c1 = csv.reader(c1, delimiter = ',')
		for f, folder in c1:
			filedict[f + folder] = f
	return filedict

def getRawfiles(rawfolder):
	"""
	Produces a list of all dialogue files
	"""
	filelist = []
	folders = [f for f in os.listdir(rawfolder)]
	for folder in folders:
		rawfiles = rawfolder + folder
		files = [f for f in os.listdir(rawfiles)]
		for f in files:
			filelist.append([f, folder])
	return filelist

def getUtterlist(c2): 
	"""
	Generates the list of utterances from the file. Can also return 'raw' list without processing.
	"""
	utterlist = []
	for row in c2:
	  row = row.split('\t')
	  if any(row[3:]):
	    utter = ''.join(row[3:])
	    utter_tok = process_line(utter)
	    utter = ' '.join(utter_tok)
	    utterlist.append(utter)
	return utterlist


def getUtterlistFromTest(context):
	return context.split(' __EOS__ ')

def makeUtterDict(testset):
	"""
	Makes a dictionary of all the contexts in the test set.
	"""
	utterdict = {}
	with open(testset, 'r') as c1:
		c1 = csv.reader(c1, delimiter = ',')
		for context, response, flag in c1:
			if int(flag) == 1:
				total_context = context + ' __EOS__ ' + response
				test_utterlist = getUtterlistFromTest(total_context)[0:3]
				test_utterlist = ' __EOS__ '.join(test_utterlist)
				utterdict[test_utterlist] = 0
	return utterdict

def findTestfiles(dialoguepath, utterdict, filelist, newtestpath, filesperprint = 100):
	newtestlist = []
	k = 0
	for folder, f in filelist:
		newpath = dialoguepath + folder + '/' + f
		with open(newpath, 'r') as c1:
			uttersplit = c1.read().split('\n')
			utterlist = getUtterlist(uttersplit)[0:3]
			new_context = ' __EOS__ '.join(utterlist)
			if new_context in utterdict:
				newtestlist.append([f, folder])
		if k % filesperprint == 0:
			print 'Finished file ' + str(k)
			writeFiles(newtestpath, newtestlist)
			newtestlist = []
		k += 1
	


badfiles = './badfiles_4.csv'
dialoguepath = './dialogs/'
testset = './testset_1.csv'


seg_index = sys.argv[1]
segfile = './dialogsegs/dialoguesegment_' + str(seg_index) + '.csv'
newtestpath = './newtestfiles_' + str(seg_index) + '.csv'

if __name__ == '__main__':
	print 'Retrieving file list'
	all_file_list = getFilesFromCsv(segfile)
	bad_file_list = getFilesFromCsv(badfiles)
	all_file_list = list(set(tuple(x) for x in all_file_list) - set(tuple(x) for x in bad_file_list))
	print 'Making testset dictionary'
	utterdict = makeUtterDict(testset)
	print 'Done initialization. Finding testfiles'
	findTestfiles(dialoguepath, utterdict, all_file_list, newtestpath)

	#all_file_list = getRawfiles(dialoguepath)
	#train_file_list = list(set(tuple(x) for x in all_file_list) - set(tuple(x) for x in test_file_list)) - set(tuple(x) for x in bad_file_list))
