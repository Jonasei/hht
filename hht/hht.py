import bchydro
import testutils
import unittest
import pandas
from pandas import *
import numpy
from numpy import *
import scipy
from scipy import *
from scipy.signal import *
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import sqlite3
import datetime
import os
from scipy.interpolate import *
from pylab import figure, show, setp
import peach.nn as p
import pickle

def removeZeroes(data):
    for i in range(len(data)):
        if data[i] == 0:
            data[i] = data[i-1]
    return data       
    


def load2():
    
    conn = sqlite3.connect('bchydro.db')
    c = conn.cursor()
    
    c.execute('SELECT MWh FROM loads')
    data2 = c.fetchall()
    
    c.close()
    return data2

 


def findLocalMaximas(input):
    maxima = []
    for i in range(1,len(input)-1):
        #if i == 0:
        #    maxima.append(i)
        #elif i == len(input)-1:
        #    maxima.append(i)
        #else:
        if input[i] > input[i+1] and input[i] > input[i-1]:
            maxima.append(i)
    return maxima
 

def findLocalMinimas(input):
    minima = []
    for i in range(1,len(input)-1):
        #if i == 0:
        #    minima.append(i)
        #elif i == len(input)-1:
        #    minima.append(i)
        #else:
        if input[i] < input[i+1] and input[i] < input[i-1]:
            minima.append(i)
    return minima

def getMaximaValues(maxima, data):
    maximaValue = []
    for i in range(len(maxima)):
        maximaValue.append(data[maxima[i]])
    return maximaValue

def getMinimaValues(minima, data):
    minimaValue = []
    for i in range(len(minima)):
        minimaValue.append(data[minima[i]])
    return minimaValue

def getMean(maximaSpline, minimaSpline):
    mean = []
    for i in range(len(maximaSpline)):
        mean.append((maximaSpline[i] + minimaSpline[i])/2)
    return mean

def getPossibleIMF(data, mean):
    IMF = []
    for i in range(len(data)):
        IMF.append(data[i]-mean[i])
    return IMF


def interpolate(xi, yi, length, order):
    s = InterpolatedUnivariateSpline(xi, yi, k=order)
    y = s(range(length))

    return y

numberOfMaxima = []
numberOfMinima = []
numberOfZeroCrossings = []
stoppingCriteriaMet = False

def differByOneOrLess(x, y):
    if x == y or x-1 == y or x == y-1:
        return True
    else:
        return False


def getMaximaSpline(maxima, maximaValues, data):
    if len(maxima) >= 4:
        maximaSpline = interpolate(maxima, maximaValues, len(data), 3)
      
    elif len(maxima) == 3:
        maximaSpline = interpolate(maxima, maximaValues, len(data), 2)
    else:
        maximaSpline = interpolate(maxima, maximaValues, len(data), 1)
    return maximaSpline

def getMinimaSpline(minima, minimaValues, data):
    if len(minima) >= 4:
        minimaSpline = interpolate(minima, minimaValues, len(data),3)
    elif len(minima) == 3:
        minimaSpline = interpolate(minima, minimaValues, len(data,),2)
    else:
        minimaSpline = interpolate(minima, minimaValues, len(data),1)
    return minimaSpline


def envlps(f):
    f = asarray(f)
    x=zeros((len(f),),dtype=int)
    y=x.copy()
    
    for i in range(1,len(f)-1):
        if (f[i]>f[i-1])&(f[i]>f[i+1]):
            x[i]=1
        if (f[i]<f[i-1])&(f[i]<f[i+1]):
            y[i]=1
    
    x=(x>0).nonzero()
    y=(y>0).nonzero()
    y=y[0]
    x=x[0]
    x=hstack((0,x,len(f)-1))
    y=hstack((0,y,len(f)-1))
   # print x
   # print type(f), f
   # print f[x]
    
    t=splrep(x,f[x])
    # numpy.arange
    top=splev(array(range(len(f))),t)
    t=splrep(y,f[y])
    bot=splev(array(range(len(f))),t)
    
    return top,bot



def isEqualToLastNTimes(list, N):
    isEqual = True
    if len(list) > N+1:
        for i in range(N):
            if list[len(list)-1] != list[len(list)-1-i]:
                isEqual = False
    else:
        isEqual = False
    return isEqual

 
def isIMF(maximas, minimas, zeroCrossings, mean,count):
    meanIsZero = True

    #for i in range(len(mean)):
    #    if (mean[i] < -6 or mean[i] > 6):
    #        meanIsZero = False
    #        break
    
    if(differByOneOrLess(maximas + minimas, zeroCrossings) and isEqualToLastNTimes(numberOfMinima, 3) and isEqualToLastNTimes(numberOfMaxima, 3) and isEqualToLastNTimes(numberOfZeroCrossings, 3)):
        return True
    else:
        #print "It is not an IMF"
        return False

def isIMF2(lastList, thisList):
    SD = 0
    for i in range(len(thisList)-1):
        increment = ((pow(abs(lastList[i]-thisList[i]), 2))/(pow(lastList[i], 2)))
        SD = SD + increment
        if i > len(thisList)-1:
            print increment
        
    return SD
def findStartDelta(list):
    delta = 0
    if list[1]-list[0]> list[2]-list[1]:
        delta = list[1]-list[0]
    else:
        delta = list[2]-list[1]
    return delta

def findEndDelta(list):
    delta = 0
    if list[len(list)-1]-list[len(list)-2]> list[len(list)-2]- list[len(list)-3]:
        delta = list[len(list)-1]-list[len(list)-2]
    else:
        delta = list[len(list)-2]-list[len(list)-3]
    return delta
    
def getListMean(list):
    sum = 0
    for value in list:
        sum += value
    mean = sum/len(list)
    return mean

def fixEndPoints(maxima, minima, data):
    #startPadding = 0
    #endPadding = 0
    #print maxima, minima
    numberOfExtraPoints = 2
    #listMean = getListMean(data)

    deltaMaxStart = findStartDelta(maxima)
    deltaMinStart = findStartDelta(minima)
    
    deltaMaxEnd = findEndDelta(maxima)
    deltaMinEnd = findEndDelta(minima)
    
    if deltaMaxStart > deltaMinStart:
        startPadding = deltaMaxStart
    else:
        startPadding = deltaMinStart
    
    if deltaMaxEnd > deltaMinEnd:
        endPadding = deltaMaxEnd
    else:
        endPadding = deltaMinEnd
        
    for j in range(numberOfExtraPoints):  
        for i in range(startPadding):
            data = numpy.insert(data,0,None)
        for i in range(endPadding):
            data = numpy.insert(data,len(data),None)
        
        maxima.insert(0,maxima[0])
        minima.insert(0,minima[0])
        
        maxima.insert(len(maxima),maxima[len(maxima)-1])
        minima.insert(len(minima),minima[len(minima)-1])

        for i in range(1,len(maxima)):
            maxima[i] = maxima[i] + startPadding
        for i in range(1,len(minima)):
            minima[i] = minima[i] + startPadding
   
        maxima[len(maxima)-1] = maxima[len(maxima)-1]+endPadding
        minima[len(minima)-1] = minima[len(minima)-1]+endPadding
    
    maxima.insert(0, 0)
    maxima.insert(len(maxima), len(data)-1)
    minima.insert(0, 0)
    minima.insert(len(minima),len(data)-1)
    
    #print maxima, minima, data, startPadding, endPadding
    return maxima, minima, data, startPadding, endPadding


plot = False
plot2 = False

def findIMF(data, count):
    print count
    maxima = findLocalMaximas(data)
    minima = findLocalMinimas(data)  
    print len(maxima), len(minima)
    

    maximaValues = getMaximaValues(maxima, data)
    minimaValues = getMinimaValues(minima, data)
    
    numberOfMaxima.append(len(maxima))
    numberOfMinima.append(len(minima))
    listMean = getListMean(data)
    
    zeroCrossings = numpy.where(numpy.sign(data[1:]) != numpy.sign(data[:-1]))
    numberOfZeroCrossings.append(len(zeroCrossings[0]))

    if not (len(maxima) >= 3 and len(minima) >= 3):
        #data = data[startPadding:-endPadding]
        print "IS DONE!"
        global done 
        done = True
        return data
    
    
    maxima, minima, data, startPadding, endPadding = fixEndPoints(maxima, minima, data)
    
    
    maximaValues.insert(0,maximaValues[1])
    maximaValues.insert(len(maximaValues), maximaValues[len(maximaValues)-2])
    minimaValues.insert(0,minimaValues[1])
    minimaValues.insert(len(minimaValues), minimaValues[len(minimaValues)-2])

    maximaValues.insert(0,maximaValues[1])
    maximaValues.insert(len(maximaValues), maximaValues[len(maximaValues)-2])
    minimaValues.insert(0,minimaValues[1])
    minimaValues.insert(len(minimaValues), minimaValues[len(minimaValues)-2])
    #print "maxmin", maxima, minima
    maximaValues.insert(0,listMean)
    maximaValues.insert(len(maximaValues), listMean)
    minimaValues.insert(0,listMean)
    minimaValues.insert(len(minimaValues), listMean)
    #print "maximaValues: ", len(maximaValues), maximaValues 
    #print "minimaValues: ", len(minimaValues), minimaValues
    
    
    #print maximaValues, minimaValues
    if len(maxima) >= 3 and len(minima) >= 3:
        maximaSpline = getMaximaSpline(maxima, maximaValues, data)
        minimaSpline = getMinimaSpline(minima, minimaValues, data)
        #maximaSpline, minimaSpline = envlps(data)
    else:
        data = data[startPadding:-endPadding]
        print "IS DONE!"
        global done 
        done = True
        return data
    #plt.plot(mean)
    #print maximaSpline, minimaSpline
    #plt.show()
 
    #print "maximaSpline: ", maximaSpline
    #print "minimaSpline: ", minimaSpline
    #plt.subplot(211)
    #plt.plot(data)
    #plt.plot(maxima, maximaValues, 'o', linestyle = 'none')
    #plt.plot(minima, minimaValues, 's', linestyle = 'none')
    #plt.plot(maximaSpline, '--')
    #plt.plot(minimaSpline, '--')   
    #lt.show()
    
    data = data[startPadding*2:-endPadding*2]
    maximaSpline = maximaSpline[startPadding*2:-endPadding*2]
    minimaSpline = minimaSpline[startPadding*2:-endPadding*2]
    mean = getMean(maximaSpline, minimaSpline)
    #mean = mean[meanOvershoot:]
    #mean = mean[startPadding:-endPadding]
    #print len(maxima), len(minima)
    
    
    
    #plt.subplot(212)
    #plt.plot(data)
    #plt.plot(data)
    #plt.plot(maximaSpline, '--')
    #plt.plot(minimaSpline, '--')
    #plt.plot(mean)
    #plt.show()
    #print len(data), len(mean)
    possibleIMF = getPossibleIMF(data, mean)
    #if len(possibleIMFs) >= 1:
    #    SD = isIMF2(possibleIMFs[len(possibleIMFs)-1], possibleIMF)
    #    print "SD", SD
        
    
    if plot:
        plt.subplot(211)
        plt.plot(mean)
        plt.plot(data)
        plt.plot(maxima, maximaValues, 'o', linestyle = 'none')
        plt.plot(minima, minimaValues, 's', linestyle = 'none')
        plt.plot(maximaSpline, '--')
        plt.plot(minimaSpline, '--')
        global plot
        plot = False
    
    print len(maxima), len(minima),len(maxima) + len(minima), len(zeroCrossings[0])
    if (isIMF(len(maxima), len(minima), len(zeroCrossings[0])+12, mean, count) or count > 900):
        
        if plot2:
            plt.subplot(212)
            plt.plot(mean)
            plt.plot(data)
            #plt.plot(maxima, maximaValues, 'o', linestyle = 'none')
            #plt.plot(minima, minimaValues, 's', linestyle = 'none')
            #plt.plot(maximaSpline, '--')
            #plt.plot(minimaSpline, '--')
            plt.grid(True)
            global plot2
            plot2 = False
            plt.show()
        
        #print "Found IMF"
        IMF = possibleIMF
        return possibleIMF
    else:
        possibleIMFs.append(possibleIMF)
        return findIMF(possibleIMF, count+1)
        
    
IMFs = []
possibleIMFs = []
done = False
testdata = []

def doHHT(database):
    global testdata
    dataset = bchydro.load(database)

    removeZeroes(dataset)

    testdata = dataset[:24*1000].values
    IMFs.append(testdata)
    #plt.figure(1)
    #plt.subplot(numberOfPlots+1,1,1)
    #plt.plot(testdata)
    

    residue = testdata
    
    #for i in range(numberOfPlots):
    while not done:
        count = 0
        #plt.subplot(5, 1, count+1)
        #plt.grid(True)
        #plt.xlabel('Time(h)')
        #plt.ylabel('MWh') 
        IMF = findIMF(residue, count)
        del possibleIMFs[:]
        IMFs.append(IMF)
        print "-------------------------------"
        residue = residue - IMF
        
        #plt.plot(IMF)
    #IMFs.append(residue)
   
  
    plt.show()
    


doHHT("bchydro.db")

def getinstfreq(imfs):
    omega=zeros((len(imfs),len(imfs[1])-1),dtype=float)
    for i in range(len(imfs)):
        h=hilbert(imfs[i:])
        theta=unwrap(angle(h))
        omega[i:]=diff(theta)
        
    return omega

omega = getinstfreq(IMFs)


def saveListToFile(list, file):
    f = open(file, 'w')
    
    pickle.dump(list, f)
    f.close()
    
saveListToFile(testdata, "testdata.txt")
saveListToFile(IMFs, "imfs.txt") 
saveListToFile(omega, "instfreq.txt")


def makePlot(list, figureNr):
    plt.figure(figureNr)
    for i in range(len(list)):
        plt.subplot(len(list),1,i+1)
        plt.plot(list[i])
        plt.xlabel("Time(h)")
        plt.ylabel("Power")
    plt.show()




makePlot(IMFs, 1)
#makePlot(omega, 2)


 


    