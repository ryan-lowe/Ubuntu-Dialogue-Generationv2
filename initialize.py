from create_parallel_files import runParFiles
from createDictionaries import createDicts

print 'Performing file segmentation'
num_segments = 1500
runParFiles(num_segments)
print 'File segmentation complete. Creating dictionaries'
createDicts()
print 'Initialization complete.'
