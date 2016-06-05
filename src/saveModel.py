#Save the learnt model to a file
#function accepts a list of layer objects and stores the objects in a file specified by the file name
from six.moves import cPickle

def saveModel(fileName, parameters):
	f = open(fileName, 'wb')
	for obj in parameters:
		cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()

#This function returns a list of layer objects
def loadModel(fileName):
	f = open(fileName, 'rb')
	loaded_objects = []
	for i in range(3):
	    loaded_objects.append(cPickle.load(f))
	f.close()
	return loaded_objects