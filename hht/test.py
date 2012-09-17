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
from scipy.interpolate import InterpolatedUnivariateSpline 
from pylab import figure, show, setp

initData = bchydro.load("bchydro.db")

def load2():
    conn = sqlite3.connect('bchydro.db')
    c = conn.cursor()
    
    c.execute('SELECT MWh FROM loads')
    data2 = c.fetchall()
    
    c.close()
    return data2




testdata = initData[0:4800].values

#print testdata

def findLocalMaximas(input):
    maxima = []
    for i in range(len(input)):
        if i == 0:
            if input[i] > input[i+1]:
                maxima.append(i)
        elif i == len(input)-1:
            if input[i] > input[i-1]:
                maxima.append(i)
        else:
            if input[i] > input[i+1] and input[i] > input[i-1]:
                maxima.append(i)
    return maxima
 

def findLocalMinimas(input):
    minima = []
    for i in range(len(input)):
        if i == 0:
            if input[i] < input[i+1]:
                minima.append(i)
        elif i == len(input)-1:
            if input[i] < input[i-1]:
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

def getIMF(data, mean):
    IMF = []
    for i in range(len(data)):
        IMF.append(data[i]-mean[i])
    return IMF

    

def interpolate(xi, yi, length):
    x = np.linspace(0,length,length)
    # spline order: 1 linear, 2 quadratic, 3 cubic ... 
    order = 3
    # do inter/extrapolation
    s = InterpolatedUnivariateSpline(xi, yi, k=order)
    y = s(x)

    return y

def runOneRound(data):
    
    
    maxima = findLocalMaximas(data)
    minima = findLocalMinimas(data)
    maximaValues = getMaximaValues(maxima, data)
    minimaValues = getMinimaValues(minima, data)
    #print data
    #print maxima
    #print minima
    #print maximaValues
    #print minimaValues

    #maximaSpline = griddata(maxima, maximaValues, range(len(data)), method = 'cubic')
    #minimaSpline = griddata(minima, minimaValues, range(len(data)), method = 'cubic')
 
    maximaSpline = interpolate(maxima, maximaValues, (len(data)))
    minimaSpline = interpolate(minima, minimaValues, len(data))
    

    mean = getMean(maximaSpline, minimaSpline)
    IMF = getIMF(data, mean)


    #plt.plot(maximaSpline)
    #plt.plot(minimaSpline)
    print "-------------------------------"


    return IMF
 


plt.figure(1)


plt.subplot(611)
IMF = runOneRound(testdata)
IMF = runOneRound(IMF)
residue = testdata - IMF
plt.plot(IMF)
 
#test  = numpy.where(numpy.sign(IMF[1:]) != numpy.sign(IMF[:-1]))


plt.subplot(612)
IMF =runOneRound(residue) 
IMF =runOneRound(IMF)
plt.plot(IMF)



plt.subplot(613)

residue = residue - IMF
IMF =runOneRound(residue)
IMF =runOneRound(IMF)
plt.plot(IMF)


plt.subplot(614)

residue = residue - IMF
IMF =runOneRound(residue)
IMF =runOneRound(IMF)
plt.plot(IMF)



plt.subplot(615)

residue = residue - IMF
IMF =runOneRound(residue)
IMF =runOneRound(IMF)
plt.plot(IMF)



plt.subplot(616)

residue = residue - IMF
IMF =runOneRound(residue)
IMF =runOneRound(IMF)
plt.plot(IMF)




plt.grid(True)
plt.xlabel('Time(h)')
plt.ylabel('MWh')


#plt.plot(mean)
#plt.plot(IMF)
plt.show()