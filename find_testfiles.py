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

def getTestfiles(testfiles):
	"""
	Produces a list of files in original testfiles.csv
	"""
	filelist = []
	with open(testfiles, 'r') as c1:
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

def makeUtterDict(filelist, path):
	"""
	Makes a dictionary of all the possible utterances, and their
	corresponding files.
	"""
	k = 0
	utterdict = {}
	for f, folder in filelist:
		foldsum = f + folder
		newpath = path + folder + '/' + f
		with open(newpath, 'r') as c1:
			utterlist = c1.read().split('\n')
			for utter in utterlist:#getUtterlist(c1):
				utterdict.setdefault(utter, set()).add(foldsum)
		k += 1
		if k % 1000 == 0:
			print k
	return utterdict

def getUtterlist(c2): 
	"""
	Generates the list of utterances from the file.
	"""
	rawutterlist = []
	for row in c2:
		row = row.split('\t')
		if any(row[3:]):
			utter = ''.join(row[3:])
			utter_tok = process_line(utter)
			utter = ' '.join(utter_tok)
			yield utter

def getUtterlistFromTest(context):
	return context.split(' __EOS__ ')


class TestFileFinder:
	def __init__(self, dialoguepath, utterdict):
		self.dialoguepath = dialoguepath
		self.utterdict = utterdict
		self.newtestlist = []

	def findFile(self, testutterlist):
		"""
		Given some list of files (file_list), and a testset question, tries to
		match question to some file in the list.
		This is done by using the dictionary or utterances. If all 
		"""
		results = None
		for utter in testutterlist:
			if utter not in self.utterdict:
				return []
			results = self.utterdict[utter] if results is None else results.intersection(self.utterdict[utter])
		for result in results:
			f = result.split('.tsv')[0] + '.tsv'
			folder = result.split('.tsv')[1]
			if self.newtestlist[-1] != [f, folder]:
				self.newtestlist.append([f, folder])
		return []

	def findTestfiles(self, testset, test_file_list, newtestpath, filesperprint = 1000):
		with open(testset, 'r') as c1:
			c1 = csv.reader(c1, delimiter = ',')
			k = 0
			for context, response, flag in c1:
				if int(flag) == 1:
					total_context = context + ' __EOS__ ' + response
					test_utterlist = getUtterlistFromTest(total_context)
					result = self.findFile(test_utterlist)
					#k += 1
					#if (k - 1) % filesperprint == 0:
					#	print 'Finished example ' + str(k)
			writeFiles(newtestpath, self.newtestlist)
					

testfiles = './testfiles.csv'
badfiles = './badfiles_4.csv'
dialoguepath = './dialogs/'
testset = './testset_1.csv'
newtestpath = './newtestfiles.csv'

if __name__ == '__main__':
	print 'Retrieving file list'
	test_file_list = getTestfiles(testfiles)
	bad_file_list = getTestfiles(badfiles)
	test_file_list = list(set(tuple(x) for x in test_file_list) - set(tuple(x) for x in bad_file_list))
	print 'Making utterance dictionary'
	utterdict = makeUtterDict(test_file_list, dialoguepath)
	test1 = TestFileFinder(dialoguepath, utterdict)
	print 'Done initialization. Finding testfiles'
	test1.findTestfiles(testset, test_file_list, newtestpath)

	#all_file_list = getRawfiles(dialoguepath)
	#train_file_list = list(set(tuple(x) for x in all_file_list) - set(tuple(x) for x in test_file_list)) - set(tuple(x) for x in bad_file_list))


"""
def findFileOld(self, testutterlist, file_list, checked_files):
	
	Given some list of files (file_list), and a testset question, tries to
	match question to some file in the list.
	If it fails, returns False.
	This is an old version that is currently not used.
	
	utterlength = len(testutterlist)
	i = 0
	while i < len(file_list):
		f, folder = file_list[i]
		badtest = f + folder
		if int(folder) >= utterlength and badtest not in self.badfile_dict and [f, folder] not in checked_files:
			filein = self.dialoguepath + folder + '/' + f
			with open(filein, 'r') as c1:
				convo_lines = c1.read().split('\n')
				utterlist = getUtterlist(convo_lines)
				#Sweeps over the utterlist to find any matches
				for j in range(len(utterlist) - utterlength - 1):
					uttertest = utterlist[j : j + utterlength]
					if testutterlist == uttertest:
						if [f, folder] not in self.newtestlist:
							self.newtestlist.append([f, folder])
							writeFiles(self.newtestpath, [[f, folder]])
							return True
		i += 1
	return False
"""
