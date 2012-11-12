import pickle
import peach as p
import matplotlib.pyplot as plt

def loadFile(fileName):
    print "Loading file: ", fileName
    file = open(fileName, 'r')
    list = pickle.load(file)
    return list

testdata = loadFile("testdata.txt")
IMFs = loadFile("imfs.txt")
instFreqs = loadFile("instfreq.txt")
grnn = p.GRNN(5)
trainingSize = 20
predictionLength = 480
trainIMFs = [1, 2, 3, 4, 5, 6]
trainFreqs = [1, 2, 3, 4, 5, 6]
time = 0

def makeTrainingList():
    nrOfParameters = len(trainIMFs)+ len(trainFreqs) + time
    trainingList = [[0 for x in range(nrOfParameters)] for x in range(trainingSize)] 
    for i in range(trainingSize):
        paramNr = 0
        for j in range(len(trainIMFs)):
            trainingList[i][paramNr]=(IMFs[trainIMFs[j]][i])
            paramNr = paramNr + 1
            
        for j in range(len(trainFreqs)):
            trainingList[i][paramNr]=(instFreqs[trainFreqs[j]][i])
            paramNr = paramNr + 1 
       # trainingList[i][3]=(instFreqs[3][i])
        if time == 1:
            trainingList[i][paramNr]= i%24
            paramNr = paramNr + 1
            
    return trainingList

def trainingGRNN():
    print "Training..."
    #trainingList = makeTrainingList()
   
    
    #grnn.train(trainingList, testdata[1:trainingSize+1])
    grnn.train(testdata[0:trainingSize], testdata[1:trainingSize+1])

def predict():
    #predictionLength = hours
    prediction = [0 for x in range(len(testdata))]
    print "Predicting..."
    predictionParams = [0 for x in range(len(trainIMFs)+len(trainFreqs)+time)]
    for i in range(trainingSize, trainingSize+predictionLength+2):
        print i
        """
        paramNr = 0
        for j in range(len(trainIMFs)):
            predictionParams[paramNr]=(IMFs[trainIMFs[j]][trainingSize+i])
            paramNr = paramNr + 1
            
        for j in range(len(trainFreqs)):
            predictionParams[paramNr]=(instFreqs[trainFreqs[j]][trainingSize+i])
            paramNr = paramNr + 1 
       # trainingList[i][3]=(instFreqs[3][i])
        if time == 1:
            predictionParams[paramNr]= (trainingSize+i)%24
            paramNr = paramNr + 1
            
        prediction[i]= grnn(predictionParams)
        """
        
        prediction[i] = grnn(testdata[i])
    return prediction
    
    
#trainingGRNN()
#print grnn
#prediction = predict()
#prediction = grnn(testdata[trainingSize+1:trainingSize+24])

"""
lprediction = []
lprediction.append(prediction[0])
for i in range(1, len(prediction)):
    a = lprediction[i-1]
    b = prediction[i]
    lprediction.append(a*0.80 + b*0.2)
"""
#plt.plot(prediction, 'r--')
plt.plot(testdata[0:240], range(240))
plt.xlabel("Hours")
plt.ylabel("MVh")
plt.show()










