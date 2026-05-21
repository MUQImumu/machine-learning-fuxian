import os
from utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


result = [range(46, 59)]

for i, subset in enumerate(result):
    
    start_chara = subset[0]
    end_chara = subset[-1] 
    
    data = np.load('../deep_learning/result_saved/random_sampling/output_all'+str(start_chara)+str(end_chara)+str(len(subset))+'.npz')
    
    R = data['R_all']
    mask = data['mask_all']
    Rhat = data['Rhat_all']
   

    for length_decile in [5]:
    
        weight, lists = get_return_weight(Rhat, R, mask, length_decile)
        
# Take a representative date and plot the portfolio weight. 
# This date responds to 2000/01. 
x = weight[12*20]

# This is prediction weight. 
y = x[1]

parameters = {'axes.labelsize': 20,
          'legend.fontsize': 12,
           'xtick.labelsize': 18,
            'ytick.labelsize':18}
plt.rcParams.update(parameters)

plt.figure(figsize = (8, 6))
# Normalize the weights back and plot. 
plt.plot(np.arange(y[y>0].shape[0]), np.sort(y[y<0])*y[y>0].shape[0], linewidth=6)
plt.plot(np.arange(y[y>0].shape[0], 2*y[y>0].shape[0]), np.sort(y[y>0])*y[y>0].shape[0], linewidth=6)
plt.xlabel('Extreme portfolios')
plt.ylabel('Predictive weight')
plt.tight_layout()
plt.savefig('plots/Figure4a.png')

## This is equal weight. 
y = x[0]

parameters = {'axes.labelsize': 20,
          'legend.fontsize': 12,
           'xtick.labelsize': 18,
            'ytick.labelsize':18}
plt.rcParams.update(parameters)

plt.figure(figsize = (8, 6))
# Normalize the weights back and plot. 
plt.plot(np.arange(y[y>0].shape[0]), y[y<0]*y[y>0].shape[0], linewidth=6)
plt.plot(np.arange(y[y>0].shape[0], 2*y[y>0].shape[0]), y[y>0]*y[y>0].shape[0], linewidth=6)
plt.xlabel('Extreme portfolios')
plt.ylabel('Equal weight')
plt.tight_layout()
plt.savefig('plots/Figure4b.png')
