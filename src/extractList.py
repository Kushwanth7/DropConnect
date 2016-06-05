import cPickle, gzip, numpy
from scipy.stats import mode
def extractListFromFile(fileName):
    f = open(fileName, 'rb')
    listExtracted = cPickle.load(f)
    return listExtracted

def processPredictedList(predictionList):
    predictedList=[]
    for i in range(len(predictionList)):
        npArr = predictionList[i]
        for j in range(npArr.shape[0]):
            predictedList.append(npArr[j])
    return predictedList

def returnErrorPercentage(actualList,predicted1,predicted2,predicted3,predicted4,predicted5):
    leng = len(predicted1)
    maxPredictions = []
    for i in range(leng):
        modeCheck = [predicted1[i],predicted2[i],predicted3[i],predicted4[i],predicted5[i]]
        maxFrequencyPredicted = mode(modeCheck)
        maxPredictions.append(maxFrequencyPredicted.mode[0])
    
    errors = 0
    for i in range(leng):
        if(maxPredictions[i] != actualList[i]):
            errors = errors + 1
   
    testError = ((errors * 1.0 )/leng) * 100
    print ("The average test error is " + str(testError))

        
    