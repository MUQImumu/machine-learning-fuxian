import os
from utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## The information sets for analysis.
result = [
    range(46, 60),\
    [59]+list(range(0, 46)), \
    range(46),\
    range(46, 59),\
]

len_result = len(result)
start_time = 0
time_factor_all = np.zeros((469,len_result))

names = []
length_decile = 10

for i, subset in enumerate(result):
    
    start_chara = subset[0]
    end_chara = subset[-1] 
    
    # Get the name of the information set. 
    name = return_name([start_chara, end_chara])        
    names.append(name)
    data = np.load('../deep_learning/result_saved/random_sampling/output_all'+ str(start_chara)+str(end_chara)+str(len(subset))+'.npz')
    
    R = data['R_all']
    mask = data['mask_all']
    Rhat = data['Rhat_all']
    
    prediction_portfolios, prediction_portfolios_timed, prediction_factor, prediction_factor_timed, real_portfolios, time_portfolios, real_factor, time_factor = get_return_time(Rhat, R, mask, length_decile)
    
    time_factor_all[:,i] = time_factor

    
time_factor_all = pd.DataFrame(time_factor_all, columns = names)
plot_decile_portfolios(time_factor_all, dateEnd='20190101', plotPath='plots/', figsize=(10, 5), name = 'Figure8',axvline = False, ylim = [-0.25, 2.0], ylabel = 'Cumulative abnormal return', legend_side = False)
   
        
