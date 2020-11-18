# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 21:30:52 2020

@author: Ana Calhau
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

x =np.arange(0.5, 10, 0.001)

y1=np.log(x)
y2=5*np.sin(x)/x

plt.plot(x,y1, label='log(x)')
plt.plot(x,y2,label='sen(x)/x')
plt.legend()

