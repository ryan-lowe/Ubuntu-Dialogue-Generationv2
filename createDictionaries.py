from __future__ import division
import math
from random import randint
from random import shuffle
import time
import os
import csv
import sysk
from nltk.corpus import wordnet as wn
from sklearn.externals import joblib
import cPickle
import re
from twokenize import tokenize



class CreateDataset:

  def __init__(self,path):
    self.timelist = []
    self.turnlist = []
    self.dictlist = []
    self.traindata = []
    self.testdata = []
    self.valdata = []

    self.traindic = {}
    self.valdic = {}
    self.testdic = {}
    self.filelist = []
    self.path = path
    self.folders = [f for f in os.listdir(self.path)]

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


global JOINSTR 
global JOIN_SENTENCE
global runscript
JOIN_SENTENCE = '. '
segments = 10
seg_index = 1

def createDicts():
  data1 = CreateDataset('./dialogs/')
  data1.createDicts(0.1)

