import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from utils import form_decile_portfolio_all, return_name, significance



"""
For this task, we use the abnormal return directly. These abnormal returns cover a larger set than the deep
learning prediction because the deep learning prediction not only requires the abnormal return to exist,
but also a large set of fund and stock characteristics. 

Since we want to get the average abnormal return from time t to t + k, we want this larger set of abnormal returns
so that we can get a more accurate estimation of long-term abnormal return. 
"""
returns_fund = pd.read_csv('../Data/abnormal_return.csv')
alpha = returns_fund.pivot_table(values='alpha', index = 'md', columns = 'wficn')
alpha = alpha.iloc[:-2].values
alpha = np.nan_to_num(alpha, nan = -99.99)
T, N = alpha.shape

result = [range(46, 60), range(60), list(range(56, 60))+[47]]

t_history = np.zeros((len(result), 36))
sharpe_history = np.zeros((len(result), 36))
mean_history = np.zeros((len(result), 36))
std_history = np.zeros((len(result), 36))

names_plot = []

for i, subset in enumerate(result):
    start_chara = subset[0]
    end_chara = subset[-1] 
    data_name = '../deep_learning/result_saved/random_sampling/output_all'+str(start_chara)+str(end_chara)+str(len(subset))+'.npz'
    data = np.load(data_name)
    model_prediction = np.expand_dims(data['Rhat_all'], axis = 2)
    names = [return_name([start_chara, end_chara])]
    names_plot.append(names[0])
    
    for m, k in enumerate(range(1, 37)):

        residual_long_period = -99.99*np.ones((T-k, N))
        for l in range(T- k):
            nansum = np.sum(alpha[l:(l+k)]==-99.99, axis = 0)
            nonzero_index = np.where(nansum==0)[0]
            residual_long_period[l, nonzero_index] = np.nansum(alpha[l:(l+k), nonzero_index], axis = 0)
        if k>0:
            model_prediction_lag = model_prediction[:-k, :, :]
        else:
            model_prediction_lag = model_prediction

        D = 10

        _, portfolio_decile_long_period = form_decile_portfolio_all(residual_long_period, model_prediction_lag, D, 1)

        start_date = 0
        end_date = 500
        output_long_period, sharpe_out_long_period = significance(portfolio_decile_long_period, names, None,start_date, end_date,  D, factor = False, sentiment_use =False, adjust_mean = 1/k, adjust_t_sharpe = 1/k, use_robust = True, lag = k, use_std = True)
        
        t_history[i,m] = sharpe_out_long_period[0][3]
        sharpe_history[i,m] = sharpe_out_long_period[0][0]
        mean_history[i,m] = sharpe_out_long_period[0][1]
        std_history[i,m] = sharpe_out_long_period[0][2]

"""
Now, plot figure 10. 
"""
plt.figure(figsize = (8, 6))
for i in range(len(result)):
    plt.plot(range(1,37), mean_history[i,:], label = names_plot[i])
plt.legend()
plt.xlabel('Number of months')
plt.ylabel('Average abnormal return')
plt.tight_layout()
plt.savefig('plots/Figure10a.png')

plt.figure(figsize = (8, 6))
for i in range(len(result)):
    plt.plot(range(1,37), sharpe_history[i,:], label = names_plot[i])
plt.xlabel('Number of months')
plt.ylabel('Sharpe ratio')
plt.tight_layout()
plt.savefig('plots/Figure10c.png')

plt.figure(figsize = (8,6))
for i in range(len(result)):
    plt.plot(range(1,37), t_history[i,:], label = names_plot[i])
plt.xlabel('Number of months')
plt.ylabel('T-statistics of abnormal mean returns')
plt.tight_layout()
plt.savefig('plots/Figure10d.png')

plt.figure(figsize = (8,6))
for i in range(len(result)):
    plt.plot(range(1,37), std_history[i,:], label = names_plot[i])
plt.xlabel('Number of months')
plt.ylabel('Std of abnormal return')
plt.tight_layout()
plt.savefig('plots/Figure10b.png')


