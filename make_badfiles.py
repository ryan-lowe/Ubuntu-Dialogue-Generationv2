from __future__ import division
import math
import csv
import sys
import re

def checkValidity(c2, percentage, convo, folder): 
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
	    writeFiles('./badfiles_'+seg_index+'.csv',[[convo, folder]])
	    return False
	return True

def getUtterlist(c2, rawtoo=False): 
	"""
	Generates the list of utterances from the file. Can also return 'raw' list without processing.
	"""
	utterlist = []
	rawutterlist = []
	for row in c2:
	  row = row.split('\t')
	  if row[0] == 'ubotu' or row[0] == 'ubottu' or row[0] == 'ubot3':
	    return [0,0] #this condition is later checked for, in order to remove ubotu dialogs
	  if len(''.join(row[3:])) != 0:
	    utter = ''.join(row[3:])
	    #if rawtoo:
	    #  rawutterlist.append(utter)
	    #utter_tok = process_line(utter)
	    #utter = ' '.join(utter_tok)
	    utterlist.append(utter)
	if rawtoo:
	  return utterlist, rawutterlist
	return utterlist

def getUserList(c2):
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

def concatUtter(utterlist,userlist):
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
	    utter = utter + '. ' + utterlist[j]
	    j += 1
	    if j == len(userlist):
	      break
	  i = j
	  utterlist_new.append(utter)
	return utterlist_new

def writeFiles(filename, data, listbool=False, overwrite=False):
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

def makeBadfiles(path, dialoguefile, seg_index, filesperprint=100, elimpct=0.2, overwrite=True):            
	"""
	Writes all files that are unused in future dataset processing steps.
	"""
	seg_index = str(seg_index)
	if overwrite:
	  writeFiles('./badfiles_'+seg_index+'.csv', [], overwrite = True)    
	k=0
	with open(dialoguefile, 'r') as dia:
	  dia = csv.reader(dia, delimiter = ',')    
	  for folder, convo in dia:
	    if int(folder) > 2: #only consider dialogs of length >= 3
	      filepath = path + folder
	      k+=1
	      if k%filesperprint == 0:
	        print 'Finished ' + str(k) + 'files'
	      filein = filepath + '/' + convo
	      with open(filein, 'r') as c1:
	        convo_lines = c1.read().split('\n')
	        utterlist, rawutterlist = getUtterlist(convo_lines, rawtoo = True)
	        userlist = getUserList(convo_lines)
	        
	        if checkValidity(convo_lines, elimpct, convo, folder):
	          utterlist = concatUtter(utterlist, userlist)
	          if len(utterlist) < 3:
	            writeFiles('./badfiles_'+seg_index+'.csv',[[convo, folder]])
	          else:
	            if utterlist[0] == utterlist[1]: #checks for ubotu utterance, and for 'good' dialogue    
	              writeFiles('./badfiles_'+seg_index+'.csv',[[convo, folder]])



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
  print 'Creating badfiles'
  makeBadfiles('./dialogs/', segfile, seg_index, overwrite=True)