import sys
import os

sys.path.append('../deep_learning/')

from src.data import data_layer_cross
from src.utils import plotIndividualFeatureImportance_cross, plotIndividualFeatureImportance_cross_group

[subset, num, hidden, dropout, max_hidden, l1_penalty, l2_penalty, lr, model_selection] = [range(46, 60), 1, [2**6], 0.95, 6, 0.0, 0.001, 0.01, 'Factor_sharpe']
start_chara = subset[0]
end_chara = subset[-1]  

directory_all = []

for i in range(3):
    directories = ['../deep_learning/output_RF/random_sampling/Train_fold_'+str(idx)+'/fullnew'+ str(start_chara)+ str(end_chara)+str(num)+ str(hidden[0]) + str(dropout)+ str(l1_penalty)+ str(l2_penalty)+ str(lr)+ model_selection+'Test'+str(i) for idx in range(1, 9)] 
    
    directory_all+= directories 

data_dir = "../deep_learning/datasets/CharAll_na_rm_huge_train_variableall4_sentiment_full_new.npz"

dl_test = data_layer_cross.DataInRamInputLayer(
    data_dir, [1], range(46, 60))


plotIndividualFeatureImportance_cross(dl_test, logdirs = directory_all,  plotPath='plots/', name = 'Figure12a',figsize=(8, 6))  
    
plotIndividualFeatureImportance_cross_group(dl_test, logdirs = directory_all,  plotPath='plots/', name = 'Figure12b',figsize=(8,6 )) 
