import os
from utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

result = [range(60), range(46, 60),  range(46)]

names = []
time_string_list = []

length_decile = 10

prefix_list = ["", "_order_", "_rolling_"]

modes = ["random_sampling", "chronological_order", "rolling_window"]

#print (">>>>>>>>>>>>>>>>>>>>>>")
#print ("Generate output for Table 5")
#print (">>>>>>>>>>>>>>>>>>>>>>")
text_file = open("tables/Table5.txt", "wt")
for idx in range(3):

    for i, subset in enumerate(result):

        start_chara = subset[0]
        end_chara = subset[-1] 

        name = return_name([start_chara, end_chara])
        names.append(name)
        data = np.load('../deep_learning/result_saved/' + modes[idx] + '/output_all'+ prefix_list[idx] + str(start_chara)+str(end_chara)+str(len(subset))+'.npz')

        R = data['R_all']
        mask = data['mask_all']
        Rhat = data['Rhat_all']
        
        if idx == 2:
            R = R[120:]
            mask = mask[120:]
            Rhat = Rhat[120:]
        
        time_string = name

        _, _, _, prediction_factor_timed, _, prediction_portfolios, _, prediction_factor = get_return_time(Rhat, R, mask, length_decile)

        factor_r_time = 1- np.sum((prediction_factor_timed-prediction_factor)**2)/np.sum(prediction_factor**2)

        x_fund_time = significance_helper(prediction_factor)

        time_string+= '&%0.2f'%(np.mean(prediction_factor)*100)+ x_fund_time[0] +'&%0.2f'%(np.mean(prediction_factor)/np.std(prediction_factor))+'&%0.2f'%(factor_r_time*100)

        time_string_list.append(time_string)


    for i in range(len(result)):
        text_file.write(time_string_list[3*idx + i]+'\\\\\n')
text_file.close()
   
        
