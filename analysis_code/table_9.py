import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

## Read in the abnormal return.
returns_fund = pd.read_csv('../Data/abnormal_return.csv')

returns_fund['risk_diff'] = returns_fund['passive_alpha'] - returns_fund['stock_alpha']
returns_fund['active_whole'] = returns_fund['active_alpha'] + returns_fund['risk_diff']

returns_fund['stock_alpha'] = returns_fund['stock_alpha'].fillna(value = -99.99)
returns_fund['active_whole'] = returns_fund['active_whole'].fillna(value = -99.99)
returns_fund['risk_diff'] = returns_fund['risk_diff'].fillna(value = -99.99)

alpha = returns_fund.pivot_table(values='alpha', index = 'md', columns = 'wficn')
alpha = alpha.iloc[:-2].values
alpha = np.nan_to_num(alpha, nan = -99.99)

alpha1 = returns_fund.pivot_table(values='stock_alpha', index = 'md', columns = 'wficn')
alpha1 = alpha1.iloc[:-2].values
alpha1 = np.nan_to_num(alpha1, nan = -99.99)

alpha2 = returns_fund.pivot_table(values='active_whole', index = 'md', columns = 'wficn')
alpha2 = alpha2.iloc[:-2].values
alpha2 = np.nan_to_num(alpha2, nan = -99.99)

alpha3 = returns_fund.pivot_table(values='risk_diff', index = 'md', columns = 'wficn')
alpha3 = alpha3.iloc[:-2].values
alpha3 = np.nan_to_num(alpha3, nan = -99.99)

alpha4 = returns_fund.pivot_table(values='active_alpha', index = 'md', columns = 'wficn')
alpha4 = alpha4.iloc[:-2].values
alpha4 = np.nan_to_num(alpha4, nan = -99.99)

result = [range(60),  range(46, 60), range(46, 59),range(59),[59]+list(range(0, 46)),range(46),list(range(56, 60))+[47]]
#print (">>>>>>>>>>>>>>>>>>>>>>")
#print ("Generate output for Table 9")
#print (">>>>>>>>>>>>>>>>>>>>>>")
text_file = open("tables/Table9.txt", "w")
for i, subset in enumerate(result):
    start_chara = subset[0]
    end_chara = subset[-1] 
    data_name = '../deep_learning/result_saved/random_sampling/output_all'+str(start_chara)+str(end_chara)+str(len(subset))+'.npz'
    data = np.load(data_name) 

    fund_chara_residual = np.expand_dims(data['Rhat_all'], axis = 2)

    names = [return_name([start_chara, end_chara])]
    D = 10
    mask = (alpha!=-99.99)
    mask= mask.astype(float)
    _, portfolio_decile = form_decile_portfolio_all(alpha, fund_chara_residual, D, 1)
    _, portfolio_decile1 = form_decile_portfolio_all(alpha1, fund_chara_residual, D, 1)
    _, portfolio_decile2 = form_decile_portfolio_all(alpha2, fund_chara_residual, D, 1)
    _, portfolio_decile3 = form_decile_portfolio_all(alpha3, fund_chara_residual, D, 1)
    _, portfolio_decile4 = form_decile_portfolio_all(alpha4, fund_chara_residual, D, 1)
    
    start_date = 0
    end_date = 500
    
    output, sharpe_out = significance(portfolio_decile, names, None,start_date, end_date,  D, factor = False, sentiment_use =False, adjust_mean = 1, adjust_t_sharpe = 1, use_robust = False, lag = 12)

    output1, sharpe_out1 = significance(portfolio_decile1, names, None,start_date, end_date,  D, sentiment_use =False, adjust_mean = 1, adjust_t_sharpe = 1, use_robust = True, lag = 12, factor = False)

    output2, sharpe_out2 = significance(portfolio_decile2, names, None,start_date, end_date,  D, factor = False, sentiment_use =False,  adjust_mean = 1, adjust_t_sharpe = 1, use_robust = True, lag = 12)

    output3, sharpe_out3 = significance(portfolio_decile3, names, None,start_date, end_date,  D, factor = False, sentiment_use =False, adjust_mean = 1, adjust_t_sharpe = 1, use_robust = True, lag = 12)

    output4, sharpe_out4 = significance(portfolio_decile4, names, None,start_date, end_date,  D, factor = False, sentiment_use =False, adjust_mean = 1, adjust_t_sharpe = 1, use_robust = True, lag = 12)
    
    ranked = np.argsort(sharpe_out)
    largest_indices = ranked[::-1]
    for k in range(1):
        text_file.write (output[k]+'&'+'&'.join(output1[k].split('&')[1:])+'&'+'&'.join(output2[k].split('&')[1:])\
           +'&'+'&'.join(output3[k].split('&')[1:])+'&'+'&'.join(output4[k].split('&')[1:])+'\\\\\n')
           
text_file.close()
