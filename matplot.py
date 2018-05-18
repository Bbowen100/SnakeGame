#!/usr/bin/env python

import numpy as np
from  matplotlib import pyplot as plt
upper  = 1000
data = np.random.random((upper, 1))
l = []
x = []
for i in range(1, (upper+1)):
    x.append(i)

for ele in data:
    l.append(ele[0])

plt.plot(x, l, 'ro')
plt.axis([0, upper, 0, 1])
plt.show()
#print(l)
