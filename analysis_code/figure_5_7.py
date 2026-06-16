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
Generate figure 5 first. 
"""   
data = np.load('../deep_learning/result_saved/random_sampling/output_all05960.npz')
R = data['R_all']
mask = data['mask_all']
Rhat = data['Rhat_all']

_, _, _, _, equal_portfolios, predictive_portfolios, _, _ = get_return_time(Rhat, R, mask, length_decile)

columns = ['Decile '+str(i) for i in range(1,length_decile+1)]

legend = True
predictive_portfolios = pd.DataFrame(predictive_portfolios, columns = columns)  
plot_decile_portfolios_all(predictive_portfolios, dateEnd='20190101', plotPath='plots/', figsize=(8,6), name = 'Figure5a', axvline = False, ylim = [-1.25, 0.85], ylabel = 'Cumulative abnormal return', legend = legend, legend_side = False) 

legend = False
equal_portfolios = pd.DataFrame(equal_portfolios, columns = columns)        
plot_decile_portfolios_all(equal_portfolios, dateEnd='20190101', plotPath='plots/', figsize=(8,6), name = 'Figure5b', axvline = False, ylim = [-1.25, 0.85], ylabel = 'Cumulative abnomal return', legend= legend, legend_side = False)


"""
Generate figure 7 next. 
"""   

filenames = ['Figure7a', 'Figure7b', 'Figure7c', 'Figure7d']

chara_list = [range(46), [59] + list(range(46)), range(46, 59), range(46, 60)]

legend = False

for i, subset in enumerate(chara_list):
    
    start_chara = subset[0]
    end_chara = subset[-1] 
    data = np.load('../deep_learning/result_saved/random_sampling/output_all'+ str(start_chara)+str(end_chara)+str(len(subset))+'.npz')
    
    R = data['R_all']
    mask = data['mask_all']
    Rhat = data['Rhat_all']

    _, _, _, _, equal_portfolios, predictive_portfolios, _, _ = get_return_time(Rhat, R, mask, length_decile)

    columns = ['Decile '+str(i) for i in range(1,length_decile+1)]

    if i == 3:
        legend = True
    predictive_portfolios = pd.DataFrame(predictive_portfolios, columns = columns)  
    plot_decile_portfolios_all(predictive_portfolios, dateEnd='20190101', plotPath='plots/', figsize=(8,6), name = filenames[i], axvline = False, ylim = [-1.25, 0.85], ylabel = 'Cumulative abnormal return', legend = legend, legend_side = False)   
