import bchydro
import testutils
import unittest
import pandas
from pandas import *
import numpy
from numpy import *
import scipy
from scipy import *
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import sqlite3
import datetime
import os
from scipy.interpolate import *
from pylab import figure, show, setp

def makePlot(list, figureNr):
    figureNr = 3
    for i in range(len(list)):
        plt.figure(figureNr)
        plt.subplot(len(list),1,i)
        plt.plot(list[i])
    plt.show()

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
    for i in range(len(input)):
        if i == 0:
            maxima.append(i)
        elif i == len(input)-1:
            maxima.append(i)
        else:
            if input[i] > input[i+1] and input[i] > input[i-1]:
                maxima.append(i)
    return maxima
 

def findLocalMinimas(input):
    minima = []
    for i in range(len(input)):
        if i == 0:
            minima.append(i)
        elif i == len(input)-1:
            minima.append(i)
        else:
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
    
    if(differByOneOrLess(maximas + minimas-4, zeroCrossings) and isEqualToLastNTimes(numberOfMinima, 3) and isEqualToLastNTimes(numberOfMaxima, 3) and isEqualToLastNTimes(numberOfZeroCrossings, 3)):
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


def findIMF(data, count):
    print count
    maxima = findLocalMaximas(data)
    minima = findLocalMinimas(data)

    zeroCrossings = numpy.where(numpy.sign(data[1:]) != numpy.sign(data[:-1]))

    numberOfMaxima.append(len(maxima))
    numberOfMinima.append(len(minima))
    numberOfZeroCrossings.append(len(zeroCrossings[0]))
    
    maximaValues = getMaximaValues(maxima, data)
    minimaValues = getMinimaValues(minima, data)

    if len(maxima) >= 4 and len(minima) >= 4:
        maximaSpline = getMaximaSpline(maxima, maximaValues, data)
        minimaSpline = getMinimaSpline(minima, minimaValues, data)
    else:
        print "IS DONE!"
        global done 
        done = True
        return data
    mean = getMean(maximaSpline, minimaSpline)

    possibleIMF = getPossibleIMF(data, mean)
    #if len(possibleIMFs) >= 1:
    #    SD = isIMF2(possibleIMFs[len(possibleIMFs)-1], possibleIMF)
    #    print "SD", SD
        
    
    
    if (isIMF(len(maxima), len(minima), len(zeroCrossings[0]), mean, count)):
       # plt.plot(mean)
       # plt.plot(data)
       # plt.plot(maxima, maximaValues, marker = 'o', linestyle = 'none')
       # plt.plot(minima, minimaValues, marker = 's', linestyle = 'none')
       # plt.plot(maximaSpline)
       # plt.plot(minimaSpline)
        #print "Found IMF"
        IMF = possibleIMF
        return possibleIMF
    else:
        possibleIMFs.append(possibleIMF)
        return findIMF(possibleIMF, count+1)
        
    
IMFs = []
possibleIMFs = []
done = False
def doHHT(database, numberOfPlots):
    
    dataset = bchydro.load(database)

    removeZeroes(dataset)

    testdata = dataset[:30000].values
    IMFs.append(testdata)
    #plt.figure(1)
    #plt.subplot(numberOfPlots+1,1,1)
    #plt.plot(testdata)
    

    residue = testdata
    
    #for i in range(numberOfPlots):
    while not done:
        #plt.subplot(numberOfPlots+1, 1, i+2)
        #plt.grid(True)
        #plt.xlabel('Time(h)')
        #plt.ylabel('MWh') 
        count = 0
        IMF = findIMF(residue, count)
        del possibleIMFs[:]
        IMFs.append(IMF)
        print "-------------------------------"
        residue = residue - IMF
        
        #plt.plot(IMF)
    #IMFs.append(residue)
   
  
    plt.show()
    

doHHT("bchydro.db", 10)
makePlot(IMFs, 1)