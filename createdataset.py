from __future__ import division
import math
from random import randint
from random import shuffle
from random import sample
from random import seed
#import random
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

seed(500)


def is_number(s):
  try:
    float(s)
    return True
  except ValueError:
    return False

def is_url(s):
    return s.startswith('http://') or s.startswith('https://') or s.startswith('ftp://') \
            or s.startswith('ftps://') or s.startswith('smb://')


def diff_times_in_seconds(t1,t2,date1,date2):
  """
  Returns the difference in time (in seconds) between two dates
  """
  t1 = t1.split(':')
  t2 = t2.split(':')
  date1 = date1.split('-')
  date2 = date2.split('-')
  if len(t1)<2 or len(t2)<2 or len(date1)<3 or len(date2)<3:
    return 60*60*24 #return 1 day if something goes wrong
  if not is_number(t1[0]) or not is_number(t1[1]) or not is_number(t2[0]) or not is_number(t2[1]):
    return 60*60*24
  if not is_number(date1[0]) or not is_number(date1[1]) or not is_number(date1[2]) or not is_number(date2[0]) \
          or not is_number(date2[1]) or not is_number(date2[2]):
    return 60*60*24
  h1,m1,s1 = int(t1[0]),int(t1[1]),0#int(t1[2])
  h2,m2,s2 = int(t2[0]),int(t2[1]),0#int(t2[2])
  d1,mo1,yr1 = int(date1[2]),int(date1[1]),int(date1[0])
  d2,mo2,yr2 = int(date2[2]),int(date2[1]),int(date2[0])
  t1_secs = s1 + 60*(m1 + 60*(h1 + 24*(d1+ 30*(mo1+12*yr1))))
  t2_secs = s2 + 60*(m2 + 60*(h2 + 24*(d2+ 30*(mo2+12*yr2))))
  return t2_secs - t1_secs


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


class CreateDataset:

  def __init__(self,path):
    """
    Initializes lists and dictionaries.
    """
    self.timelist = []
    self.turnlist = []
    self.wordlist = []
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

    self.testfakes = []
    self.valfakes = []
    self.trainfakes = []
    self.rawtestfakes = []
    self.testfakecount = 0
    self.trainfakecount = 0

    self.worddict = {}


  def getUtterlist(self, c2, rawtoo=False): 
    """
    Generates the list of utterances from the file. Can also return 'rww' list without processing.
    """
    utterlist = []
    rawutterlist = []
    for row in c2:
      row = row.split('\t')
      if row[0] == 'ubotu' or row[0] == 'ubottu' or row[0] == 'ubot3':
        return [0,0]
      if len(''.join(row[3:])) != 0:
        utter = ''.join(row[3:])
        if rawtoo:
          rawutterlist.append(utter)
        utter_tok = process_line(utter)
        utter = ' '.join(utter_tok)
        utterlist.append(utter)
    if rawtoo:
      if len(utterlist) != len(rawutterlist):
        print 'UTTERLIST IS:'
        print utterlist
        print 'RAWUTTERLIST IS:'
        print rawutterlist
      return utterlist, rawutterlist
    return utterlist

  def getUserList(self, c2):
    """
    Returns the list of users forom a given conversation.
    """
    userlist = []
    for row in c2:
      row = row.split('\t')
      if row[0] == 'ubotu' or row[0] == 'ubottu' or row[0] == 'ubot3':
        return [0,0]
      if len(''.join(row[3:])) != 0:
        userlist.append(row[1])
    return userlist  

  def checkValidity(self, c2, percentage, convo): 
    """
    Checks whether we accept or reject a given conversation.
    """
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
        self.writeFiles('./deletedfiles.csv', [convo])
    return True

  def wordSim(self, fake_response, real_response, context=[]):
    """
    Calculates the tf-idf score of two responses.
    """
    fake_response = fake_response.split(' ')
    real_response = real_response.split(' ')
    if context != []:
      context = context.split(' ')
    count = 0.0
    for word in fake_response:
      for word2 in real_response:
        if word == word2 and word in self.worddict:
          count += 1.0 / (math.log(self.worddict[word]) + 1.0)
        elif word == word2:
          count += 1.0
      for word2 in context:
        if word == word2 and word in self.worddict:
          count += 1.0 / (math.log(self.worddict[word]) + 1.0)
        elif word == word2:
          count += 1.0
    return count / (len(fake_response) + len(real_response))

  def makeTimeList(self, c2):
    """
    Return a list of the times in a dialogue.
    """
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

  def generateResponses(self, num_responses, convo, testpct, real_response=None, context=None, 
                        random=True, regen_fakes=False, fakelist=None):
    """
    Generates the fake responses for each question.
    Can be either randomly selected, or with tf-idf.
    """
    fakes = []
    rawfakes =[]
    i = 0
    if random:
      """
      Code for generating responses randomly.
      """
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
          utterlist, raw_utterlist = self.getUtterlist(c1, rawtoo = True)
          random_index = randint(0, len(utterlist)-1)
          fake = utterlist[random_index]
          rawfake = raw_utterlist[random_index]
          if isinstance(fake, basestring) == False or isinstance(rawfake, basestring) == False: #check if it is a string
            break
          if len(fake) > 1 and len(rawfake) > 1:
            fakes.append(fake)
            rawfakes.append(rawfake)
            i += 1
    else:
      """
      Code for generating responses using tf-idf.
      """
      fakescores = []
      if regen_fakes: #regen_fakes indicates if you regenerate the false responses stored in memory
        num_searched = 120 #number of false responses stored in memory at a time
        fakefiles = []
        """
        First randomly select files from train, test, or validation dictionaries.
        """
        if convo in self.traindic:
          nums = sample(range(0, int(len(self.filelist)*(1-2*testpct))-1), num_searched)
          for i in range(num_searched):
            fakefiles.append(self.path + self.filelist[nums[i]][1] + '/' + self.filelist[nums[i]][0])
        elif convo in self.valdic:
          nums = sample(range(int(len(self.filelist)*(1-2*testpct)), int(len(self.filelist)*(1-testpct))), num_searched)
          for i in range(num_searched):
            fakefiles.append(self.path + self.filelist[nums[i]][1] + '/' + self.filelist[nums[i]][0])
        else:
          nums = sample(range(int(len(self.filelist)*(1-testpct)), len(self.filelist)-1), num_searched)
          for i in range(num_searched):
            fakefiles.append(self.path + self.filelist[nums[i]][1] + '/' + self.filelist[nums[i]][0])
        for files in fakefiles:
          with open(files, 'r') as c1:
            if fakelist == self.testfakes:
              utterlist1, utterlist2 = self.getUtterlist(c1, rawtoo=True)
              for i in range(len(utterlist1)):
                utterwords = utterlist1[i].split(' ')
                if isinstance(utterlist1[i], basestring) and len(utterlist2[i]) > 5 and utterlist1[i] != real_response:
                  if not (utterwords < 3 and any(word in STOPWORDS for word in utterwords)):
                    fakelist.append(utterlist1[i])
                    self.rawtestfakes.append(utterlist2[i])
            else:
              utterlist = self.getUtterlist(c1)
              for utter in utterlist:
                if isinstance(utter, basestring) and len(utter) > 1:
                  fakelist.append(utter)
      for fake in fakelist:
        fakescores.append(self.wordSim(fake, real_response, context))
      fakeindex = sorted(range(len(fakescores)), key=lambda k: -fakescores[k])
      if len(self.testfakes) > len(self.rawtestfakes):
        self.testfakes = self.testfakes[0:len(self.rawtestfakes)]
      elif len(self.rawtestfakes) > len(self.testfakes):
        self.rawtestfakes = self.rawtestfakes[0:len(self.testfakes)]
      j = 0
      k = 0
      while j < num_responses and k < len(fakeindex):
        #print fakeindex[k]
        #print len(fakelist)
        #print len(self.rawtestfakes)
        if fakelist == self.testfakes:
          if fakeindex[k] < len(fakelist) and fakeindex[k] < len(self.rawtestfakes):
            if fakelist[fakeindex[k]] not in fakes:
              fakes.append(fakelist[fakeindex[k]])
              rawfakes.append(self.rawtestfakes[fakeindex[k]])
              j += 1
        else:
          if fakeindex[k] < len(fakelist):
            if fakelist[fakeindex[k]] not in fakes:
              fakes.append(fakelist[fakeindex[k]])
              j += 1
        k += 1
    if fakelist == self.testfakes:
      return fakes, rawfakes
    return fakes

  def createDicts(self, testpct, trainfiles = None, valfiles = None, testfiles = None):
    """
    Assigns dialogeus to either the training, validation, or test sets.
    'Trainfiles' indicates a list that has already been made and saved.
    """
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
        self.writeFiles('./trainfiles.csv', [self.filelist[i]])
      for i in xrange(int(len(self.filelist)*(1-2*testpct)), int(len(self.filelist)*(1-testpct))):
        self.traindic[self.filelist[i][0] + self.filelist[i][1]] = self.filelist[i][1]
        self.writeFiles('./valfiles.csv', [self.filelist[i]])
      for i in xrange(int(len(self.filelist)*(1-testpct)), len(self.filelist)):
        self.testdic[self.filelist[i][0] + self.filelist[i][1]] = self.filelist[i][1]  
        self.writeFiles('./testfiles.csv', [self.filelist[i]])
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
    """
    Writes to .csv files (overwrite optional).
    """
    csvname = filename
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
    """
    Concatenates consecutive utterances by the same user.
    """
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

  def makeBadfiles(c2, filein, folder):
    """
    Makes a list of files that are not used.
    """
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
      self.writeFiles('./badfiles.csv', [[filein, folder]])  
  
  def makeWordDict(self):
    """
    Make a dictionary of words that are used.
    """
    num_searched = 500
    searchfiles = []
    wordlist = []
    nums = sample(range(0, int(len(self.filelist)-1)), num_searched)
    for i in range(num_searched):
      searchfiles.append(self.path + self.filelist[nums[i]][1] + '/' + self.filelist[nums[i]][0])
    for files in searchfiles:
      with open(files, 'r') as c1:
        utterlist = self.getUtterlist(c1)
        for utters in utterlist:
          words = utters.split(' ')
          for word in words:
            wordlist.append(word)
    for word in wordlist:
      if word not in self.worddict:
        self.worddict[word] = 1
      else:
        self.worddict[word] += 1

  def appendTrainData(self, utterlist, check_dict, max_context_size, convo, testpct, num_options_train, random=True):
    """
    Given utterlist, calls generateResponses, appends context and responses to training data.
    """
    perfakeregen = 1000
    for i in xrange(2, len(utterlist) - 1):
      context = utterlist[max(0, i - max_context_size):i]
      context = JOINSTR.join(context)  
      response = utterlist[i]
      if random:
        fakes = self.generateResponses(num_options_train - 1, check_dict, testpct)
      else:
        if self.trainfakecount%perfakeregen == 0:
          self.trainfakecount += 1
          fakes = self.generateResponses(num_options_train - 1, check_dict, testpct, real_response=response, context=context, 
                                        regen_fakes=True, fakelist=self.trainfakes, random = False)
        else:
          fakes = self.generateResponses(num_options_train - 1, check_dict, testpct, real_response=response, context=context, 
                                        fakelist=self.trainfakes, random = False)
      context_words = context.split(' ')
      if len(context_words) > 5:
        data = [[context, response, 1]]
        for fake in fakes:
          data.append([context, fake, 0])
        self.traindata.append(data)

  def appendTestData(self, utterlist, check_dict, max_context_size, convo, testpct, num_options_test, datatype, 
                    faketype, random=True, rawutterlist=None):
    """
    Given utterlist, calls generateResponses, appends context and responses to test or validation data.
    """
 
    perfakeregen = 2000
    rawfakes = []
    contextsize = int((max_context_size*10) / randint(max_context_size/2, max_context_size*10)) + 2
    if contextsize > len(utterlist):
      contextsize = len(utterlist)
    for i in xrange(0, int((len(utterlist))/contextsize)):
      j = i*contextsize
      context = utterlist[j:j + contextsize - 1]
      #print context
      context = JOINSTR.join(context)  
      response = utterlist[j + contextsize - 1]
      if rawutterlist != None:
        rawcontext = rawutterlist[j:j + contextsize - 1]
       # print "RAWCONTEXT IS:"
       # print rawcontext
        rawcontext = JOINSTR.join(rawcontext)  
        rawresponse = rawutterlist[j + contextsize - 1]

      if random:
        if datatype == self.testdata:
          fakes, rawfakes = self.generateResponses(num_options_test - 1, check_dict, testpct, fakelist = faketype)
        else:
          fakes = self.generateResponses(num_options_test - 1, check_dict, testpct, fakelist = faketype)
          if len(fakes[0]) >= 2:
            if len(fakes[0][0]) > 1:
              fakes = fakes[0]
      else:
        """
        Check whether you should regenerate your list of false responses.
        Not done at every step for computational efficiency.
        """
        if self.testfakecount%perfakeregen == 0: 
          self.testfakecount += 1
          if datatype == self.testdata:
            fakes, rawfakes = self.generateResponses(num_options_test - 1, check_dict, testpct, real_response = response, 
                                                    context = context, regen_fakes = True, fakelist = faketype, random = False)
          else:
            fakes = self.generateResponses(num_options_test - 1, check_dict, testpct, real_response = response, context = context, 
                                          regen_fakes = True, fakelist = faketype, random = False)
        else:
          if datatype == self.testdata:
            fakes, rawfakes = self.generateResponses(num_options_test - 1, check_dict, testpct, real_response = response, 
                                                    context = context, fakelist = faketype, random = False)
          else:
            fakes = self.generateResponses(num_options_test - 1, check_dict, testpct, real_response = response, context = context, 
                                          fakelist = faketype, random = False)
      context_words = context.split(' ')
      if len(context_words) > 5:
        data = [[context, response, 1]]  
        for fake in fakes:              
          data.append([context, fake, 0]) 
        datatype.append(data)
        if rawutterlist != None:
          rawdata = [[rawcontext, rawresponse, 1]]  
          for rawfake in rawfakes:              
            rawdata.append([rawcontext, rawfake, 0]) 
          self.rawtestdata.append(rawdata)
                  

  def sortFilesParallel(self, dialoguefile, seg_index, max_context_size=20, num_options_train=2, num_options_test=2, 
                        testpct=0.1, filesperprint=100, elimpct=0.2, badfiles=False, overwrite=True, random = True):            
    """
    Given that the dialogues have been sorted with CreateDicts, constructs the training, test,
    and validation files.
    """
    print 'Constructing word dictionary'
    if random == False:
      self.makeWordDict()
    print 'Finished construction'

    seg_index = str(seg_index)
    if overwrite:
      self.writeFiles('./trainset_'+seg_index+'.csv', [], overwrite = True)
      self.writeFiles('./valset_'+seg_index+'.csv', [], overwrite = True)        
      self.writeFiles('./testset_'+seg_index+'.csv', [], overwrite = True)   
      self.writeFiles('./rawtestset_'+seg_index+'.csv', [], overwrite = True)
      self.writeFiles('./turndata_'+seg_index+'.csv', [], overwrite = True)       
      self.writeFiles('./badfiles_'+seg_index+'.csv', [], overwrite = True)    
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
            utterlist, rawutterlist = self.getUtterlist(c2, rawtoo = True)
            userlist = self.getUserList(c2)

            if  badfiles: #for adding syntax stuff to badfiles.csv   
              makeBadfiles(c2, filein, folder)                      
            
            if self.checkValidity(c2, elimpct, convo):
              utterlist = self.concatUtter(utterlist, userlist)
              rawutterlist = self.concatUtter(rawutterlist, userlist)
              if len(utterlist) < 3:
                self.writeFiles('./badfiles_'+seg_index+'.csv',[[convo, folder]])
              else:
                if utterlist[0] != utterlist[1]: #checks for ubotu utterance, and for 'good' dialogue           
                  self.turnlist.append(len(utterlist))
                  utterwordlist = utterlist
                  for utter in utterwordlist:
                    utter = utter.split(' ')
                    self.wordlist.append(len(utter))
                  self.makeTimeList(c2)
                  check_dict = convo + folder
                  if check_dict in self.traindic:
                    self.appendTrainData(utterlist, check_dict, max_context_size, convo, testpct, num_options_train, random = random)
                    self.dictlist.append(0)
                  elif check_dict in self.valdic:
                    self.appendTestData(utterlist, check_dict, max_context_size, convo, testpct, num_options_test, 
                                        self.valdata, self.valfakes, random = random)  
                    self.dictlist.append(1)   
                  else:
                    self.appendTestData(utterlist, check_dict, max_context_size, convo, testpct, num_options_test, 
                                        self.testdata, self.testfakes, random = random, rawutterlist = rawutterlist)
                    self.dictlist.append(0)  
                    i += 1
              if k % filesperprint == 0 or folder + convo == lastfold:
                if self.traindata != []:
                  self.writeFiles('./trainset_'+seg_index+'.csv', self.traindata, listbool=True)
                if self.valdata != []:
                  self.writeFiles('./valset_'+seg_index+'.csv', self.valdata, listbool = True)
                if self.testdata != []:
                  self.writeFiles('./testset_'+seg_index+'.csv', self.testdata, listbool=True)
                  self.writeFiles('./rawtestset_'+seg_index+'.csv',self.rawtestdata, listbool=True)
                for i in range(len(self.timelist)):
                  self.writeFiles('./turndata_'+seg_index+'.csv', [(self.timelist[i], self.turnlist[i], 
                                  self.wordlist[i], self.dictlist[i])])             
                self.traindata = []
                self.valdata = []
                self.testdata = []
                self.rawtestdata = []
                self.timelist = []
                self.turnlist = []
                self.dictlist = []
                self.wordlist = []


global JOINSTR 
global JOIN_SENTENCE
global STOPWORDS
JOINSTR = ' __EOS__ ' #string that you use to join utterances
JOIN_SENTENCE = '. ' #string that you use to join sentences
STOPWORDS = ['thanks','thank','ok','okay']
segments = 10


seg_index = sys.argv[1]
segfile = './dialogsegs/dialoguesegment_' + str(seg_index) + '.csv'
"""
Input has to be of the form:
segfile = './dialogsegs/dialoguesegment_0.csv'
"""

if __name__ == "__main__":
  """
  Calls functions that create the dataset.
  Note: createDicts accepts as inputs the pre-separated train, test, and validation files.
  If starting from scratch, remove these from function call.

  Current Settings:
  Adds dialogue to 'badfiles' if there's more than 5 utterances, and one user has less
  than 20% of the utterances.
  """
  data1 = CreateDataset('./dialogs/')
  data1.createDicts(0.02, trainfiles = './trainfiles.csv', valfiles = './valfiles.csv', testfiles = './testfiles.csv')
  print 'Finished dictionaries, making data files'
  data1.sortFilesParallel(segfile, seg_index, num_options_test=10, overwrite=True, testpct=0.02)


