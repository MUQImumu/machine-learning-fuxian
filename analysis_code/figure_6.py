import os
from utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


"""
Print the cumulative plot for expenses.
"""

result = [range(60), range(60)]

length_decile = 10
    
## Get additional information on expense ratio and net abnormal return.
residual_data = pd.read_csv('../Data/abnormal_return.csv')

for i, subset in enumerate(result):
    
    start_chara = subset[0]
    end_chara = subset[-1] 
    
    # Load the saved deep learning predictions. 
    data = np.load('../deep_learning/result_saved/random_sampling/output_all'+str(start_chara)+str(end_chara)+str(len(subset))+'.npz')
    residual = data['residual_all']
    
    # Get the original fund id that the deep learning is predicting. 
    data_temp = np.load('../deep_learning/datasets/CharAll_na_rm_huge_train_variableall4_sentiment_full_new.npz')
    funds_main = data_temp["wficn"]
    
    if i == 0:
        ## Look at data between 1980/01 to 2019/01. 
        alpha = residual_data.pivot_table(values='exp_ratio', index = 'Date', columns = 'wficn').iloc[:-2]   
    else:
        ## Look at data between 1980/01 to 2019/01. 
        alpha = residual_data.pivot_table(values='net_alpha', index = 'Date', columns = 'wficn').iloc[:-2]

    ### Because some data are downloaded at later times, the set of funds
    ### are different, and I need to make sure that the set of funds studied are intersecting. 
    
    funds_auxiliary = list(alpha.columns)    
    funds_intersection = [x for x in funds_main if x in funds_auxiliary]    
    funds_auxiliary = [i for i in range(len(funds_auxiliary)) if funds_auxiliary[i] in funds_intersection]
    funds_main = [i for i in range(len(funds_main)) if funds_main[i] in funds_intersection]   
    alpha = alpha.values[:, funds_auxiliary]
    R = np.nan_to_num(alpha, nan = -99.99)    
    mask = (data['mask_all'][:, funds_main])*(R!=-99.99).astype(float)    
    Rhat = data['Rhat_all'][:, funds_main]       
    
    _, _, _,_,  _, prediction_portfolios , _, _= get_return_time(Rhat, R, mask, length_decile)

    columns = ['Decile '+str(i) for i in range(1,11)]
    columns[0]+=' (Bottom)'
    columns[9]+=' (Top)'
    
    prediction_portfolios = pd.DataFrame(prediction_portfolios, columns = columns)        
    if i== 0:
        plot_decile_portfolios_all(prediction_portfolios, dateEnd='20190101', plotPath='plots/', figsize=(8,6), 
                                   name = 'Figure6a', ylim = [0,0.55], \
                                   ylabel = 'Cumulative expense ratio', axvline = False, legend = False) 
    else: 
        plot_decile_portfolios_all(prediction_portfolios, dateEnd='20190101', plotPath='plots/', figsize=(8,6), 
                                   name = 'Figure6b', ylim = [-1.75,0.75], \
                                   ylabel = 'Cumulative abnomal net return', axvline = True, legend_side = False) 
        
    
        
