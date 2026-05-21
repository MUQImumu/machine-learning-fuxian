import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
This is the code for generating figure 1 and figure 2. 


"""



parameters = {'axes.labelsize': 18,
          'legend.fontsize': 12,
           'xtick.labelsize': 16,
            'ytick.labelsize':16}
plt.rcParams.update(parameters)
"""
This is the code for generating Figure 1. 

For easy processing of the data, we define an variable "md",
which is the 
"""
macro_data = pd.read_csv('../Data/macro_combined.csv')
macro_states_sen = macro_data.loc[(macro_data['md']>=685-36) & (macro_data['md']<= 1117), 'Sentiment'].values
macro_states_sen = macro_states_sen.astype(float)

columns = ['RecCFNAI']
macro_data = pd.read_csv('../Data/macro_combined.csv')
macro_states_rec = macro_data.loc[(macro_data['md']>=685-36) & (macro_data['md']<= 1117), 'CFNAI'].values
macro_states_rec = macro_states_rec.astype(float)

plt.figure(figsize=(12,4))
plt.plot(pd.date_range(end = '12/1/2018', periods = 469,  freq= 'M'), macro_states_sen, label = 'Sentiment', color = 'steelblue')
plt.tight_layout()
plt.savefig('plots/Figure1a.png')


plt.figure(figsize=(12,4))
plt.plot(pd.date_range(end = '12/1/2018', periods = 469,  freq= 'M'), macro_states_rec, label = 'CFNAI', color = 'steelblue')
plt.tight_layout()
plt.savefig('plots/Figure1b.png')


"""
This is the code for generating figure 2. 
"""

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'tab:blue', 'tab:purple', 'tab:orange', 'tab:brown']    
df = pd.DataFrame(macro_states_sen, columns = ['Sentiment'])
date = pd.date_range(end='20181201', periods=macro_states_sen.shape[0], freq='MS')
df.loc[:,'date'] = date
df.set_index('date', inplace=True)
fig = plt.figure(figsize=(12,4))
final_list = np.load('../deep_learning/sampling_folds/random_sampling_folds.npy', allow_pickle = True)
for k, decile in enumerate(df.columns):
	s = df.loc[:,decile]
	if decile=='Sentiment':
		for i in range(3):
			[train_idx_list, valid_idx_list, test_idx_list] = final_list[i]             
			plt.scatter(s.index[test_idx_list], s.iloc[test_idx_list], color = colors[i], s = 5.0)        
	plt.tight_layout()
	plt.savefig(('plots/Figure2a.png'))
# 	plt.show()
fig = plt.figure(figsize=(12,4))
final_list = np.load('../deep_learning/sampling_folds/chronological_order_folds.npy', allow_pickle = True)
for k, decile in enumerate(df.columns):
	s = df.loc[:,decile]
	if decile=='Sentiment':
		for i in range(3):
			[train_idx_list, valid_idx_list, test_idx_list] = final_list[i]             
			plt.scatter(s.index[test_idx_list], s.iloc[test_idx_list], color = colors[i], s = 5.0)        
	plt.tight_layout()
	plt.savefig(('plots/Figure2b.png'))
# 	plt.show()
    
    
