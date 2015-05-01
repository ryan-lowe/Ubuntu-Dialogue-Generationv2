from __future__ import division
import math
from random import randint
from random import shuffle
import time
import os
import csv
import sys
import nltk
from nltk.corpus import wordnet as wn
from sklearn.externals import joblib
import cPickle
import re
from twokenize import tokenize

#optimization that is currently not used
def read_random_line(f, chunk_size=128): 
    import random
    with open(f, 'rb') as f_handle:
        f_handle.seek(0, os.SEEK_END)
        size = f_handle.tell()
        i = random.randint(0, size)
        while True:
            i -= chunk_size
            if i < 0:
                chunk_size += i
                i = 0
            f_handle.seek(i, os.SEEK_SET)
            chunk = f_handle.read(chunk_size)
            i_newline = chunk.rfind(b'\n')
            if i_newline != -1:
                i += i_newline + 1
                break
            if i == 0:
                break
        f_handle.seek(i, os.SEEK_SET)
        return f_handle.readline()



def is_number(s):
  try:
    float(s)
    return True
  except ValueError:
    return False


def diff_times_in_seconds(t1,t2,date1,date2):
  t1 = t1.split(':')
  t2 = t2.split(':')
  date1 = date1.split('-')
  date2 = date2.split('-')
  if len(t1)<2 or len(t2)<2 or len(date1)<3 or len(date2)<3:
    return 60*60*24 #return 1 day if something goes wrong
  if not is_number(t1[0]) or not is_number(t1[1]) or not is_number(t2[0]) or not is_number(t2[1]):
    return 60*60*24
  if not is_number(date1[0]) or not is_number(date1[1]) or not is_number(date1[2]) or not is_number(date2[0]) or not is_number(date2[1]) or not is_number(date2[2]):
    return 60*60*24
  h1,m1,s1 = int(t1[0]),int(t1[1]),0#int(t1[2])
  h2,m2,s2 = int(t2[0]),int(t2[1]),0#int(t2[2])
  d1,mo1,yr1 = int(date1[2]),int(date1[1]),int(date1[0])
  d2,mo2,yr2 = int(date2[2]),int(date2[1]),int(date2[0])
  t1_secs = s1 + 60*(m1 + 60*(h1 + 24*(d1+ 30*(mo1+12*yr1))))
  t2_secs = s2 + 60*(m2 + 60*(h2 + 24*(d2+ 30*(mo2+12*yr2))))
  return t2_secs - t1_secs


def is_url(s):
    return s.startswith('http://') or s.startswith('https://') or s.startswith('ftp://') or s.startswith('ftps://') or s.startswith('smb://')


def replace_sentence(text):
    if isinstance(text,basestring) == False:
      return text
    words = nltk.word_tokenize(text)
    sent = nltk.pos_tag(words)
    chunks = nltk.ne_chunk(sent, binary=False)
    sentence = []
    nodelist = ['PERSON','ORGANIZATION','GPE','LOCATION','FACILITY','GSP']
    for c,word in zip(chunks, words):
        changed = False
        if hasattr(c, 'node'):     
            if c.node in nodelist:
                sentence.append("__%s__" % c.node) 
                changed = True
        if not changed:
          if is_url(word):
              sentence.append("__URL__")
          elif is_number(word):
              sentence.append("__NUMBER__")
          elif os.path.isabs(word):
              sentence.append("__PATH__")
          else:
            sentence.append(word)           
    return " ".join(sentence)            

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
    string = string.replace('</s>', '__EOS__')
    return string.strip() 

def process_token(c, word):
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
    if clean_string:
        s = clean_str(s)
    tokens = tokenize(s)
    #return [process_token(None,token).lower() for token in tokens]
    sent = nltk.pos_tag(tokens)
    chunks = nltk.ne_chunk(sent, binary=False)
    return [process_token(c,token).lower().encode('UTF-8') for c,token in map(None, chunks, tokens)]


class CreateDataset:

  def __init__(self,path):
    self.timelist = []
    self.turnlist = []
    self.dictlist = []
    self.traindata = []
    self.testdata = []
    self.rawtestdata = []
    self.valdata = []

    self.traindic = {}
    self.valdic = {}
    self.testdic = {}
    self.filelist = []
    self.path = path
    self.folders = [f for f in os.listdir(self.path)]

  #generates the list of utterances from the file
  def getUtterlist(self, c2): 
    utterlist = []
    for row in c2:
      row = row.split('\t')
      if row[0] == 'ubotu' or row[0] == 'ubottu' or row[0] == 'ubot3':
        return [0,0]
      if len(''.join(row[3:])) != 0:
        utter = ''.join(row[3:])
        utter_tok = process_line(utter)

        utter = ' '.join(utter_tok)
        utterlist.append(utter)
    return utterlist

  def getRawUtterlist(self, c2): 
    utterlist = []
    for row in c2:
      row = row.split('\t')
      if row[0] == 'ubotu' or row[0] == 'ubottu' or row[0] == 'ubot3':
        return [0,0]
      if len(''.join(row[3:])) != 0:
        utter = ''.join(row[3:])
        utterlist.append(utter)
    return utterlist

  def getUserList(self, c2):
    userlist = []
    for row in c2:
      row = row.split('\t')
      if row[0] == 'ubotu' or row[0] == 'ubottu' or row[0] == 'ubot3':
        return [0,0]
      if len(''.join(row[3:])) != 0:
        userlist.append(row[1])
    return userlist  

  #checks whether we accept or reject a file
  def checkValidity(self, c2, percentage, convo): 
    userlist = []
    uniqueuser = {}
    for row in c2:
      row = row.split('\t')
      if len(row) > 1:
        if len(row[1]) != 0:
          userlist.append(row[1])
          if row[1] not in uniqueuser:
            uniqueuser[row[1]] = 1
          else:
            uniqueuser[row[1]] += 1
    for user,value in uniqueuser.iteritems():
      if value < percentage*len(userlist) and len(userlist) > 5:
        return False
        self.writeFiles('../deletedfiles.csv', [convo])
    return True


  def makeTimeList(self, c2):
    firstind = 0
    firstval = c2[0].split('\t')[0]
    while len(firstval.split('T')) < 2:
      firstind += 1
      firstval = c2[firstind].split('\t')[0]
    lastind = -2
    lastval = c2[-2].split('\t')[0]
    while len(lastval.split('T')) < 2:
      lastind -= 1
      lastval = c2[lastind].split('\t')[0]    
    firstdate = firstval.split('T')[0]
    firsttime = firstval.split('T')[1].split('Z')[0]
    lastdate = lastval.split('T')[0]
    lasttime = lastval.split('T')[1].split('Z')[0]
    timediff = diff_times_in_seconds(firsttime, lasttime, firstdate, lastdate)
    self.timelist.append(timediff)    

  def generateResponses(self, num_responses, convo, testpct):
    fakes = []
    i = 0
    while i < num_responses:
      if convo in self.traindic:
        num = randint(0, int(len(self.filelist)*(1-2*testpct))-1)
        fakefile = self.path + self.filelist[num][1] + '/' + self.filelist[num][0]
      elif convo in self.valdic:
        num = randint(int(len(self.filelist)*(1-2*testpct)), int(len(self.filelist)*(1-testpct)))
        fakefile = self.path + self.filelist[num][1] + '/' + self.filelist[num][0]
      else:
        num = randint(int(len(self.filelist)*(1-testpct)), len(self.filelist)-1)
        fakefile = self.path + self.filelist[num][1] + '/' + self.filelist[num][0]
      with open(fakefile, 'r') as c1:
        utterlist = self.getUtterlist(c1)
        c2 = utterlist[randint(0, len(utterlist)-1)]
        if isinstance(c2,basestring) == False: #check if it is a string
          break
        if len(c2) > 1:
          fakes.append(c2)
          i += 1
    return fakes

  def createDicts(self, testpct, trainfiles = None, valfiles = None, testfiles = None):
    print 'Creating dictionary of training, validation, and test sets'
    if trainfiles == None:
      for folder in self.folders:
        if int(folder) > 2:
          filepath = self.path + folder
          for f in os.listdir(filepath):
            self.filelist.append([f, folder])
      shuffle(self.filelist)
      for i in xrange(int(len(self.filelist)*(1-2*testpct))):
        self.traindic[self.filelist[i][0] + self.filelist[i][1]] = self.filelist[i][1]
        self.writeFiles('../trainfiles.csv', [self.filelist[i]])
      for i in xrange(int(len(self.filelist)*(1-2*testpct)), int(len(self.filelist)*(1-testpct))):
        self.traindic[self.filelist[i][0] + self.filelist[i][1]] = self.filelist[i][1]
        self.writeFiles('../valfiles.csv', [self.filelist[i]])
      for i in xrange(int(len(self.filelist)*(1-testpct)), len(self.filelist)):
        self.testdic[self.filelist[i][0] + self.filelist[i][1]] = self.filelist[i][1]  
        self.writeFiles('../testfiles.csv', [self.filelist[i]])
    else:
      with open(trainfiles, 'r') as c1:
        c1 = csv.reader(c1, delimiter = ',')
        for f,folder in c1:
          self.filelist.append([f, folder])
          self.traindic[f + folder] = folder
      with open(valfiles, 'r') as c1:
        c1 = csv.reader(c1, delimiter = ',')
        for f,folder in c1:
          self.filelist.append([f, folder])
          self.valdic[f + folder] = folder        
      with open(testfiles, 'r') as c1:
        c1 = csv.reader(c1, delimiter = ',')
        for f,folder in c1:
          self.filelist.append([f, folder])
          self.testdic[f + folder] = folder  
    

  def writeFiles(self, filename, data, listbool=False, overwrite=False):
    csvname = self.path + filename
    if overwrite:
     with open(csvname,'w') as out:
      csv_out = csv.writer(out)
      for row in data:
        if listbool:
          for col in row:
            csv_out.writerow(col)
        else:
          csv_out.writerow(row)
    else:
      with open(csvname,'a+') as out:
        csv_out = csv.writer(out)
        for row in data:
          if listbool:
            for col in row:
              csv_out.writerow(col)
          else:
            csv_out.writerow(row)
  
  def concatUtter(self, utterlist,userlist):
    utterlist_new = []
    i = 0
    while i < len(utterlist):
      utter = utterlist[i]
      if i == len(utterlist) - 1:
        utterlist_new.append(utter)
        break
      j = i+1
      while userlist[i] == userlist[j] and j < len(userlist):
        utter = utter + JOIN_SENTENCE + utterlist[j]
        j += 1
        if j == len(userlist):
          break
      i = j
      utterlist_new.append(utter)
    return utterlist_new

  def makeBadfiles(c2, filein):
    utterlist = []
    namedict = {}
    for row in c2:
      row = row.split('\t')
      if len(''.join(row[3:])) != 0:
        utterlist.append(''.join(row[3:]))
      if len(row) < 4 and len(row[0]) != 0:
        namedict['error'] = 0
      if len(row) > 3:
        if len(row[2]) != 0:
          namedict[row[2]] = 0
        namedict[row[1]] = 0
        if len(row[1]) == 0:
          namedict['error'] = 0
    if len(namedict) > 2:
      self.writeFiles('../badfiles.csv', [[filein]])  



  def appendTrainData(self, utterlist, check_dict, max_context_size, convo, testpct, num_options_train):
    for i in xrange(2, len(utterlist) - 1):
      context = utterlist[max(0, i - max_context_size):i]
      context = JOINSTR.join(context)  
      response = utterlist[i]
      fakes = self.generateResponses(num_options_train - 1, check_dict, testpct)  
      data = [[context, response, 1]]
      for fake in fakes:
        data.append([context, fake, 0])
      self.traindata.append(data)

  def appendTestData(self, utterlist, check_dict, max_context_size, convo, testpct, num_options_test, val=None):
    contextsize = int((max_context_size*10) / randint(max_context_size/2, max_context_size*10)) + 2
    if contextsize > len(utterlist):
      contextsize = len(utterlist)
    for i in xrange(0, int((len(utterlist))/contextsize)):
      j = i*contextsize
      context = utterlist[j:j + contextsize - 1]
      context = JOINSTR.join(context)  
      response = utterlist[j + contextsize - 1]
      fakes = self.generateResponses(num_options_test - 1, check_dict, testpct)  
      data = [[context, response, 1]]  
      for fake in fakes:              
        data.append([context, fake, 0]) 
      if val==1:
        self.valdata.append(data)
      elif val==0:
        self.testdata.append(data)
      else:
        self.rawtestdata.append(data)
      #self.writeFiles('../testfiles.csv', [[convo,contextsize-1]])   
"""
  def sortFiles(self, max_context_size=20, num_options_train=2, num_options_test=2, testpct=0.1, filesperprint=100, elimpct=0.2, badfiles=False):            
    firstline = [['Context','Response','Correct']]
    self.writeFiles('../trainset.csv', [], overwrite = True)
    self.writeFiles('../valset.csv', [], overwrite = True)        
    self.writeFiles('../testset.csv', [], overwrite = True)   
    self.writeFiles('../turnlist.csv', [], overwrite = True)       
    self.writeFiles('../timelist.csv', [], overwrite = True)       
    self.writeFiles('../badfiles.csv', [], overwrite = True)    
    
    for folder in self.folders:     
      if int(folder) > 2:
        print '   Starting ' + folder + ' folder'
        filepath = self.path + folder
        files = [f for f in os.listdir(filepath)]
        k=0
        i=0
        for convo in files:
          k+=1
          if k%100 == 0:
            print 'Finished ' + str(k) + 'files'
          filein = filepath + '/' + convo
          with open(filein, 'r') as c1:
            c2 = c1.read().split('\n')
            utterlist = self.getUtterlist(c2)
            userlist = self.getUserList(c2)

            if  badfiles: #for adding syntax stuff to badfiles.csv   
              makeBadfiles(c2, filein)                      
            
            if self.checkValidity(c2, elimpct, convo):
              utterlist = self.concatUtter(utterlist, userlist)
              if len(utterlist) < 3:
                self.writeFiles('../badfiles.csv',[[convo]])
              else:
                if utterlist[0] != utterlist[1]: #checks for ubotu utterance, and for 'good' dialogue           
                  self.turnlist.append(len(utterlist))
                  self.makeTimeList(c2)
                  check_dict = convo + folder
                  if check_dict in self.traindic:
                    self.appendTrainData(utterlist, check_dict, max_context_size, convo, testpct, num_options_train)
                    self.dictlist.append(0)
                  elif check_dict in self.valdic:
                    self.appendTestData(utterlist, check_dict, max_context_size, convo, testpct, num_options_test, val=True)     
                    self.dictlist.append(1)
                  else:
                    self.appendTestData(utterlist, check_dict, max_context_size, convo, testpct, num_options_test, val=False)     
                    self.dictlist.append(2)
                    i+=1
              if i % 10 == 0:#k % filesperprint == 0 or k == len(files):
                if self.traindata != []:
                  self.writeFiles('../trainset.csv', self.traindata, listbool=True)
                if self.valdata != []:
                  self.writeFiles('../valset.csv', self.valdata, listbool = True)
                if self.testdata != []:
                  self.writeFiles('../testset.csv', self.testdata, listbool=True)
                self.writeFiles('../turndata.csv', [(self.timelist, self.turnlist, self.dictlist)])  
                self.traindata = []
                self.valdata = []
                self.testdata = []
                self.timelist = []
                self.turnlist = []
              
"""
  def sortFilesParallel(self, dialoguefile, seg_index, max_context_size=20, num_options_train=2, num_options_test=2, testpct=0.1, filesperprint=100, elimpct=0.2, badfiles=False, overwrite=False):            
    #firstline = [['Context','Response','Correct']]
    seg_index = str(seg_index)
    if overwrite:
      self.writeFiles('../trainset_'+seg_index+'.csv', [], overwrite = True)
      self.writeFiles('../valset_'+seg_index+'.csv', [], overwrite = True)        
      self.writeFiles('../testset_'+seg_index+'.csv', [], overwrite = True)   
      self.writeFiles('../turndata_'+seg_index+'.csv', [], overwrite = True)       
      self.writeFiles('../badfiles_'+seg_index+'.csv', [], overwrite = True)    
    k=0
    i=0
    with open(dialoguefile, 'r') as dia:
      dia = csv.reader(dia, delimiter = ',')
      for folder, convo in dia:
        lastfold = folder + convo
    with open(dialoguefile, 'r') as dia:
      dia = csv.reader(dia, delimiter = ',')    
      for folder, convo in dia:
        if int(folder) > 2:
          filepath = self.path + folder
          k+=1
          if k%filesperprint == 0:
            print 'Finished ' + str(k) + 'files'
          filein = filepath + '/' + convo
          with open(filein, 'r') as c1:
            c2 = c1.read().split('\n')
            utterlist = self.getUtterlist(c2)
            userlist = self.getUserList(c2)
            rawutterlist = self.getRawUtterlist(c2)

            if  badfiles: #for adding syntax stuff to badfiles.csv   
              makeBadfiles(c2, filein)                      
            
            if self.checkValidity(c2, elimpct, convo):
              utterlist = self.concatUtter(utterlist, userlist)
              if len(utterlist) < 3:
                self.writeFiles('../badfiles_'+seg_index+'.csv',[[convo]])
              else:
                if utterlist[0] != utterlist[1]: #checks for ubotu utterance, and for 'good' dialogue           
                  self.turnlist.append(len(utterlist))
                  self.makeTimeList(c2)
                  check_dict = convo + folder
                  if check_dict in self.traindic:
                    self.appendTrainData(utterlist, check_dict, max_context_size, convo, testpct, num_options_train)
                    self.dictlist.append(0)
                  elif check_dict in self.valdic:
                    self.appendTestData(utterlist, check_dict, max_context_size, convo, testpct, num_options_test, val=1)  
                    self.dictlist.append(1)   
                  else:
                    self.appendTestData(utterlist, check_dict, max_context_size, convo, testpct, num_options_test, val=0)
                    self.appendTestData(rawutterlist, check_dict, max_context_size, convo, testpct, num_options_test, val=-1)   
                    self.dictlist.append(0)  
                    i += 1
              if k % filesperprint == 0 or folder + convo == lastfold:
                if self.traindata != []:
                  self.writeFiles('../trainset_'+seg_index+'.csv', self.traindata, listbool=True)
                if self.valdata != []:
                  self.writeFiles('../valset_'+seg_index+'.csv', self.valdata, listbool = True)
                if self.testdata != []:
                  self.writeFiles('../testset_'+seg_index+'.csv', self.testdata, listbool=True)
                  self.writeFiles('../rawtestset_'+seg_index+'.csv',self.rawtestdata, listbool=True)
                for i in range(len(self.timelist)):
                  self.writeFiles('../turndata_'+seg_index+'.csv', [(self.timelist[i], self.turnlist[i], self.dictlist[i])])             
                self.traindata = []
                self.valdata = []
                self.testdata = []
                self.timelist = []
                self.turnlist = []
                self.dictlist = []
              """
              if folder + convo == lastfold:
                if self.traindata != []:
                  self.writeFiles('../trainset_'+seg_index+'.csv', self.traindata, listbool=True)
                if self.valdata != []:
                  self.writeFiles('../valset_'+seg_index+'.csv', self.valdata, listbool = True)
                if self.testdata != []:
                  self.writeFiles('../trainset_'+seg_index+'.csv', self.testdata, listbool=True)
                for i in range(len(self.timelist)):
                  self.writeFiles('../turndata_'+seg_index+'.csv', [(self.timelist[i], self.turnlist[i], self.dictlist[i])])             
                self.traindata = []
                self.valdata = []
                self.testdata = []
                self.timelist = []
                self.turnlist = []
                self.dictlist = []
              """


global JOINSTR 
global JOIN_SENTENCE
global runscript
JOINSTR = '__EOS__'
JOIN_SENTENCE = '. '
segments = 10


seg_index = sys.argv[1]
segfile = './dialogsegs/dialoguesegment_' + str(seg_index) + '.csv'
#segfile = sys.argv[1]
"""
Input has to be of the form:
segfile = './dialogsegs/dialoguesegment_0.csv'
"""

def createNewDicts():
  data2 = CreateDataset('./dialogs/')
  data2.createDicts(0.1)

def runScript():
  data1 = CreateDataset('./dialogs/')
  data1.createDicts(0.1, trainfiles = './trainfiles.csv', valfiles = './valfiles.csv', testfiles = './testfiles.csv')
  print 'Finished dictionaries, making data files'
  #data1.sortFiles(num_options_test=10)
  data1.sortFilesParallel(segfile, seg_index, num_options_test=10, overwrite=True)

runScript()

"""
adds to badfiles if utters > 5 and less than 20% of utterances


"""