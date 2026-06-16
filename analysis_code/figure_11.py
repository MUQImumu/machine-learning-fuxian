import os
from utils import get_return_time
from utils import plot_decile_portfolios_all
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
Parameters of the plot
"""
length_decile = 10

"""
Generate penal a first. 
"""   

filenames = ['Figure11a1', 'Figure11a2', 'Figure11b1', 'Figure11b2']

chara_list = [range(46, 60), range(46), range(46, 60), range(46)]

legend = False

for i, subset in enumerate(chara_list):
    
    start_chara = subset[0]
    end_chara = subset[-1] 
    
    if i <= 1:
        data = np.load('../deep_learning/result_saved/chronological_order/output_all_order_'+ str(start_chara)+str(end_chara)+str(len(subset))+'.npz')
    else:
        data = np.load('../deep_learning/result_saved/rolling_window/output_all_rolling_'+ str(start_chara)+str(end_chara)+str(len(subset))+'.npz')
    
    R = data['R_all']
    mask = data['mask_all']
    Rhat = data['Rhat_all']
    
    if i > 1: 
        R = R[120:]
        mask = mask[120:]
        Rhat = Rhat[120:]

    _, _, _, _, equal_portfolios, predictive_portfolios, _, _ = get_return_time(Rhat, R, mask, length_decile)

    columns = ['Decile '+str(i) for i in range(1,length_decile+1)]

    legend = True
    predictive_portfolios = pd.DataFrame(predictive_portfolios, columns = columns)  
    plot_decile_portfolios_all(predictive_portfolios, dateEnd='20190101', plotPath='plots/', figsize=(8,6), name = filenames[i], axvline = False, ylim = [-1.5, 1.5], ylabel = 'Cumulative abnormal return', legend = legend, legend_side = False)   
