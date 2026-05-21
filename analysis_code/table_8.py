import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

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

data_name = '../Data/alpha_imputed_chara_sentiment.npz'
data = np.load(data_name)

fund_chara_residual = data['data'][:,:,1:-1]
sentiment = data['data'][:,0,-1]
names =  data['names']

D = 10
mask = (alpha!=-99.99)
mask= mask.astype(float)

portfolio_decile, _ = form_decile_portfolio_all(alpha, fund_chara_residual, D)

portfolio_decile1, _ = form_decile_portfolio_all(alpha1, fund_chara_residual, D)

portfolio_decile2, _ = form_decile_portfolio_all(alpha2, fund_chara_residual, D)

portfolio_decile3, _ = form_decile_portfolio_all(alpha3, fund_chara_residual, D)

portfolio_decile4,_ = form_decile_portfolio_all(alpha4, fund_chara_residual, D)


start_date = 0
end_date = 500
output, sharpe_out = significance(portfolio_decile, names, sentiment,start_date, end_date,  D, factor = False, sentiment_use =False, adjust_mean = 1, adjust_t_sharpe = 1, use_robust = False, lag = 12)

output1, sharpe_out1 = significance(portfolio_decile1, names, sentiment,start_date, end_date,  D, factor = False, sentiment_use =False, adjust_mean = 1, adjust_t_sharpe = 1, use_robust = True, lag = 12)

output2, sharpe_out2 = significance(portfolio_decile2, names, sentiment,start_date, end_date,  D, factor = False, sentiment_use =False, adjust_mean = 1, adjust_t_sharpe = 1, use_robust = True, lag = 12)

output3, sharpe_out3 = significance(portfolio_decile3, names, sentiment,start_date, end_date,  D, factor = False, sentiment_use =False, adjust_mean = 1, adjust_t_sharpe = 1, use_robust = True, lag = 12)

output4, sharpe_out4 = significance(portfolio_decile4, names, sentiment,start_date, end_date,  D, factor = False, sentiment_use =False, adjust_mean = 1, adjust_t_sharpe = 1, use_robust = True, lag = 12)

_, _, _, sharpe_out = zip(*sharpe_out)

ranked = np.argsort(sharpe_out)
largest_indices = ranked[::-1]
#print (">>>>>>>>>>>>>>>>>>>>>>")
#print ("Generate output for Table 8")
#print (">>>>>>>>>>>>>>>>>>>>>>")
text_file = open("tables/Table8.txt", "w")
for k in largest_indices[:15]:
    text_file.write (output[k]+'&'+'&'.join(output1[k].split('&')[1:])+'&'+'&'.join(output2[k].split('&')[1:])\
           +'&'+'&'.join(output3[k].split('&')[1:])+'&'+'&'.join(output4[k].split('&')[1:])+'\\\\\n')
text_file.close()
