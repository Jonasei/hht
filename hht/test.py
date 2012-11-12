from numpy import *
import peach.nn as p

grnn = p.GRNN()

grnn.train([[1,2,3],[3,4,5]],[2, 10])


print grnn([2,2,2])

"""

from scipy import *
from scipy.signal import *

import numpy as np
import matplotlib.pyplot as plt
import bitarray
from peach import import n
bitarray.test()


t2 = np.arange(0.0, 5.0, 0.01)
sine = np.sin(2*np.pi*t2)
cos = np.cos(2*np.pi*t2)
plt.plot(sine)


def getinstfreq(imfs):
    omega=zeros((len(imfs)-1),dtype=float)
    for i in range(len(imfs)):
        h=hilbert(imfs[i:])
        theta=unwrap(angle(h))
        omega[i:]=diff(theta)
        
    return omega


x = getinstfreq(t2)
plt.plot(x)
plt.plot(cos)
plt.show()
"""