import os
from utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

result = [range(60), range(46, 60),range(46, 59),  range(59), [59]+list(range(0, 46)), range(46), 
          [58, 59], list(range(56, 60))+[47], [59]+ [x for x in range(46, 58) if x not in (list(range(54, 58))+[47])]]

names = []
time_string_list = []

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
    
    _, _, _, prediction_factor_prediction, _, _, _, prediction_factor = get_return_time(Rhat, R, mask, length_decile)
      
    factor_r_time = 1- np.sum((prediction_factor_prediction-prediction_factor)**2)/np.sum(prediction_factor**2)    
    x_fund_time = significance_helper(prediction_factor)

    time_string+= '&%0.2f'%(np.mean(prediction_factor)*100)+ x_fund_time[0] +'&%0.2f'%(np.mean(prediction_factor)/np.std(prediction_factor))+'&%0.2f'%(factor_r_time*100)
    
    time_string_list.append(time_string)

#print (">>>>>>>>>>>>>>>>>>>>>>")
#print ("Generate output for Table 3")
#print (">>>>>>>>>>>>>>>>>>>>>>")
text_file = open("tables/Table3.txt", "wt")

for i in range(len(result)):
    text_file.write(time_string_list[i]+'\\\\\n')
text_file.close()
   
        
