import sys
import os
sys.path.append('../deep_learning/')

from src.data import data_layer_cross
from src.utils import plotconditionalmean_cross


[subset, num, hidden, dropout, max_hidden, l1_penalty, l2_penalty, lr, model_selection] = [range(46, 60), 1, [2**6], 0.95, 6, 0.0, 0.001, 0.01, 'natural']
start_chara = subset[0]
end_chara = subset[-1]  

data_dir = "../deep_learning/datasets/CharAll_na_rm_huge_train_variableall4_sentiment_full_new.npz"

dl_test = data_layer_cross.DataInRamInputLayer(
    data_dir, [1], range(46, 60))

plot_list = ['F_r12_2','flow', 'F_ST_Rev', 'Family_r12_2' ]

plot_names = ["Figure13a", "Figure13b", "Figure13c", "Figure13d"]

for k_num, k_name in enumerate(plot_list):
    if k_num==1:
        legend = True
    else:
        legend = False
    plotconditionalmean_cross(dl_test, 4, k_name, 'sentiment', plotPath='plots/', figsize=(8,6), name = plot_names[k_num], legend = legend, cross_idx_num = 3, base_dir = "../deep_learning/result_saved/random_sampling/Interaction/")
