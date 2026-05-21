import os
from utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

result = [[47, 58, 59]]

names = []
time_string_list = []
time_top_list = []
time_bottom_list = []

length_decile = 10

for i, subset in enumerate(result):
    
    start_chara = subset[0]
    end_chara = subset[-1] 

    name = return_name([start_chara, end_chara])
    names.append(name)
    data = np.load('../deep_learning/result_saved/random_sampling/output_all'+ str(start_chara)+str(end_chara)+str(len(subset))+'.npz')
    
    R = data['R_all']
    mask = data['mask_all']
    Rhat = data['Rhat_all']
    time_string = name
    top_time_string = name
    bottom_time_string = name
    
    _, prediction_portfolios_timed, _, prediction_factor_timed, _, prediction_portfolios, _, prediction_factor = get_return_time(Rhat, R, mask, length_decile)
      
    factor_r_time = 1- np.sum((prediction_factor_timed-prediction_factor)**2)/np.sum(prediction_factor**2)    
    top_r_time = 1 - np.sum((prediction_portfolios_timed[:,-1] - prediction_portfolios[:,-1])**2)/np.sum(prediction_portfolios[:,-1]**2)
    bottom_r_time = 1 - np.sum((prediction_portfolios_timed[:,0] - prediction_portfolios[:,0])**2)/np.sum(prediction_portfolios[:,0]**2) 
    
    
    x_fund_time = significance_helper(prediction_factor)    
    x_fund_top_time = significance_helper(prediction_portfolios[:,-1])
    x_fund_bottom_time = significance_helper(prediction_portfolios[:,0])

    time_string+= '&%0.2f'%(np.mean(prediction_factor)*100)+ x_fund_time[0] +'&%0.2f'%(np.mean(prediction_factor)/np.std(prediction_factor))+'&%0.2f'%(factor_r_time*100)
    top_time_string += '&%0.2f'%(np.mean(prediction_portfolios[:,-1])*100)+ x_fund_top_time[0] +'&%0.2f'%(np.mean(prediction_portfolios[:,-1])/np.std(prediction_portfolios[:,-1]))+'&%0.2f'%(top_r_time*100)   
    bottom_time_string += '&%0.2f'%(np.mean(prediction_portfolios[:,0])*100)+ x_fund_bottom_time[0] +'&%0.2f'%(np.mean(prediction_portfolios[:,0])/np.std(prediction_portfolios[:,-1]))+'&%0.2f'%(bottom_r_time*100)
    
    time_string_list.append(time_string)
    time_top_list.append(top_time_string)
    time_bottom_list.append(bottom_time_string)
#print (">>>>>>>>>>>>>>>>>>>>>>")
#print ("Generate output for Table 7")
#print (">>>>>>>>>>>>>>>>>>>>>>")
text_file = open("tables/Table7.txt", "wt")

for i in range(len(result)):
    text_file.write(time_string_list[i]+'\\\\\n')
for i in range(len(result)):
    text_file.write(time_top_list[i]+'\\\\\n')
for i in range(len(result)):
    text_file.write(time_bottom_list[i]+'\\\\\n')
text_file.close()
        
