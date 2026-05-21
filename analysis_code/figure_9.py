import os
from utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
This is the code for generating the value weighted version of the fund abnormal return. 

The input used is just abnormal returns and the tna of mutual funds. 
"""

result = [range(46, 60)]
factor_all = np.zeros((469,1))
length_decile = 10
plot_names = ["Figure9a", "Figure9b"]

for idx in range(2):
    for i, subset in enumerate(result):

        start_chara = subset[0]
        end_chara = subset[-1] 

        data = np.load('../deep_learning/result_saved/random_sampling/output_all'+str(start_chara)+str(end_chara)+str(len(subset))+'.npz')
        residual = data['residual_all']

        #### First, get the number of funds.
        T, N = residual.shape

        data_temp = np.load('../deep_learning/datasets/CharAll_na_rm_huge_train_variableall4_sentiment_full_new.npz')
        list_rret = list(data_temp['wficn'])
        returns_fund = pd.read_csv("../Data/abnormal_return.csv")
        returns_fund = returns_fund.dropna(subset = ['md'])
        ### Drop funds smaller than a threshold.
        if idx == 1:
            returns_fund = returns_fund[returns_fund['tna']>= 15]
        value_fund = returns_fund.pivot_table(values='tna', index = 'md', columns = 'wficn').iloc[:-2]

        list_ret = list(value_fund.columns)

        list_intersection = [x for x in list_rret if x in list_ret]

        idx_ret = [i for i in range(len(list_ret)) if list_ret[i] in list_intersection]

        idx_rret = [i for i in range(len(list_rret)) if list_rret[i] in list_intersection]

        value_fund = value_fund.values[:,idx_ret]

        R = data['R_all'][:, idx_rret]

        mask = data['mask_all'][:, idx_rret]*(~np.isnan(value_fund))
        Rhat = data['Rhat_all'][:, idx_rret] 
        columns = ['Decile '+str(i) for i in range(1,11)]
        columns[0]+=' (Bottom)'
        columns[9]+=' (Top)'
        
        if idx == 0:
            prediction_portfolios, prediction_portfolios_timed, prediction_factor, prediction_factor_timed, equal_portfolios, prediction_portfolios, equal_factor, prediction_factor = get_return_time(Rhat, R, mask, length_decile, value_fund)
            equal_portfolios = pd.DataFrame(equal_portfolios, columns = columns)        
            plot_decile_portfolios_all(equal_portfolios, dateEnd='20190101', plotPath='plots/', figsize=(8, 6), name = plot_names[idx], axvline = False, ylim = [-1.25, 1], ylabel = 'Cumulative abnomal return', legend_side = False) 
        else:
            prediction_portfolios, prediction_portfolios_timed, prediction_factor, prediction_factor_timed, equal_portfolios, prediction_portfolios, equal_factor, prediction_factor = get_return_time(Rhat, R, mask, length_decile)

#             prediction_portfolios = pd.DataFrame(prediction_portfolios, columns = columns)        
#             plot_decile_portfolios_all(prediction_portfolios, dateEnd='20190101', plotPath='plots/', figsize=(8, 6), name = plot_names[idx], axvline = False, ylim = [-1.25, 1], ylabel = 'Cumulative abnomal return', legend_side = False) 
            equal_portfolios = pd.DataFrame(equal_portfolios, columns = columns)        
            plot_decile_portfolios_all(equal_portfolios, dateEnd='20190101', plotPath='plots/', figsize=(8, 6), name = plot_names[idx], axvline = False, ylim = [-1.25, 1], ylabel = 'Cumulative abnomal return', legend_side = False) 
