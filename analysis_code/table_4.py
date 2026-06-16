import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

start_month = 198001
end_month = 201901

variable_use = 'alpha'

factors_8 = pd.read_csv('../Data/8_factors.csv')
names_list = np.arange(47, 60)

idx_start = np.where(factors_8['Date']==start_month)[0][0]
idx_end = np.where(factors_8['Date']==end_month)[0][0]
ff4 = factors_8.iloc[idx_start:(idx_end+1)]

data = np.load('../deep_learning/datasets/CharAll_na_rm_huge_train_variableall4_sentiment_full_new.npz')

residual_data = pd.read_csv('../Data/abnormal_return.csv')

alpha = residual_data.loc[residual_data['wficn'].isin(list(data['wficn']))].pivot_table(values='rret', index = 'Date', columns = 'wficn').iloc[:-2].values

alpha = np.nan_to_num(alpha, nan = -99.99)

names_print = []

D = 10
mask = (alpha!=-99.99)
mask= mask.astype(float)

columns_list = [[ 'Mkt-RF', 'SMB', 'HML', 'Mom'], \
               ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'] , \
                [ 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom'], \
               ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'LT_Rev', 'Mom', 'ST_Rev']]


result = [range(60), range(46, 60), range(46, 59),range(59),  [59]+list(range(0, 46)),  range(46), 
          list(range(56, 60))+[47], [58, 59]]

correlation = np.zeros((len(columns_list)*2, len(result)))
significance = np.zeros((len(columns_list), len(result)))
std_errs = np.zeros((len(columns_list), len(result)))

mean_factor = np.zeros((len(result)))
std_factor = np.zeros(len(result))
p_factor = np.zeros(len(result))


for i, subset1 in enumerate(result):
    start_chara1 = subset1[0]
    end_chara1 = subset1[-1]
    data_name = '../deep_learning/result_saved/random_sampling/output_all'+str(start_chara1)+str(end_chara1)+str(len(subset1))+'.npz'
    data = np.load(data_name)

    fund_chara_residual = data['Rhat_all']
    
    mask = data['mask_all']

    name = return_name([start_chara1, end_chara1])
    names_print.append(name)
    
    prediction_portfolios, prediction_portfolios_timed, prediction_factor, prediction_factor_timed, \
    real_portfolios, time_portfolios, real_factor, factor = get_return_time(fund_chara_residual, \
                   alpha, mask, D)

    factor = factor*100
    factor = factor/np.std(factor, axis = 0)
    for k, columns in enumerate(columns_list):
        X = sm.add_constant(ff4.loc[:,columns])
        mod = sm.OLS(factor,X)
        fii = mod.fit()
        mean = fii.summary2().tables[1]['Coef.']['const']
        t_stats = fii.summary2().tables[1]['t']['const']
        p_values = fii.summary2().tables[1]['P>|t|']['const']
        std_err =  fii.summary2().tables[1]['Std.Err.']['const']
        correlation[2*k, i], significance[k,i], correlation[2*k+1, i], std_errs[:,i] = mean, p_values, fii.rsquared, std_err
        X = np.ones((factor.shape[0]))
        mod = sm.OLS(factor,X)
        fii = mod.fit()
        mean = fii.summary2().tables[1]['Coef.']['const']
        std_err =  fii.summary2().tables[1]['Std.Err.']['const']
        p_value =  fii.summary2().tables[1]['P>|t|']['const']
        mean_factor[i], std_factor[i], p_factor[i] = mean, std_err, p_value
        
#print (">>>>>>>>>>>>>>>>>>>>>>")
#print ("Generate output for Table 4")
#print (">>>>>>>>>>>>>>>>>>>>>>")
text_file = open("tables/Table4.txt", "wt")
print_regression_many(correlation.T,significance.T, names_print+['R2'], \
                 ['Intercept', '$R^2$', 'Intercept', \
                  '$R^2$', 'Intercept', '$R^2$', 'Intercept', '$R^2$', 'Factor mean'], [], False, std_errs.T, True,\
                      mean_factor, std_factor, p_factor, print_factor = True, orders = range(correlation.shape[1]), filename = text_file)
text_file.close()
