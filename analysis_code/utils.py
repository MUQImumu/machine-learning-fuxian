import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

def get_return_time(Rhat, R, mask, length_decile, values = None):
    T = mask.shape[0]
    prediction_portfolio = np.zeros((T, length_decile))
    prediction_portfolio_timing = np.zeros((T, length_decile))    
    equal_portfolio = np.zeros((T, length_decile)) 
    prediction_portfolio = np.zeros((T, length_decile))
    lists = []    
    for t in range(T):
        nonzero_index = np.where(mask[t]!=0)[0]      
        rank = np.argsort(Rhat[t][nonzero_index])              
        num_decile = len(nonzero_index)//length_decile
        num_decile_left = len(nonzero_index) - num_decile*(length_decile-1)        
        for i in range(length_decile):
            if i==length_decile-1:
                predicted_returns_weight = Rhat[t][nonzero_index[rank[-num_decile:]]].copy()
                predicted_returns = Rhat[t][nonzero_index[rank[-num_decile:]]]  
                returns= R[t][nonzero_index[rank[-num_decile:]]]
                equal_portfolio[t, i] = np.mean(returns)                
                if values is not None:
                    values_temp = values[t][nonzero_index[rank[-num_decile:]]]
                    equal_portfolio[t, i] = np.sum(returns*values_temp)/np.sum(values_temp)   
            else:
                predicted_returns_weight = Rhat[t][nonzero_index[rank[i*num_decile:(i+1)*num_decile]]].copy()
                predicted_returns = Rhat[t][nonzero_index[rank[i*num_decile:(i+1)*num_decile]]]                  
                returns= R[t][nonzero_index[rank[i*num_decile:(i+1)*num_decile]]]               
                equal_portfolio[t, i] = np.mean(returns) 
                if values is not None:
                    values_temp = values[t][nonzero_index[rank[i*num_decile:(i+1)*num_decile]]]                   
                    equal_portfolio[t, i] = np.sum(returns*values_temp)/np.sum(values_temp)          
            # For the     
            if i>=length_decile//2:
                predicted_returns_weight -= np.min(predicted_returns)
            if i<length_decile//2:
                predicted_returns_weight -= np.max(predicted_returns)   
    
            prediction_portfolio[t, i] = \
            np.mean(predicted_returns)
            prediction_portfolio_timing[t, i] = \
            np.sum(predicted_returns*predicted_returns_weight)/np.sum(predicted_returns_weight)
                
            prediction_portfolio[t,i] = np.sum(returns*predicted_returns_weight)/np.sum(predicted_returns_weight) 
            if values is not None:
                prediction_portfolio[t,i] = np.sum(returns*predicted_returns_weight*values_temp)/np.sum(predicted_returns_weight*values_temp)      
    return prediction_portfolio, prediction_portfolio_timing, prediction_portfolio[:,-1]- prediction_portfolio[:,0], prediction_portfolio_timing[:,-1]- prediction_portfolio_timing[:,0], equal_portfolio, prediction_portfolio, equal_portfolio[:,-1]- equal_portfolio[:, 0], prediction_portfolio[:, -1] - prediction_portfolio[:, 0]


def plot_decile_portfolios_all(df, n_train=240, n_valid=60, dateEnd='20161201',  plotPath=None, figsize=(8,6), name = '', axvline= True, ylabel = 'Cumulative expense ratio', ylim = [-0.4,0.4], legend = True, legend_side = True):
    date = pd.date_range(end=dateEnd, periods=df.shape[0], freq='MS')
    df.loc[:,'date'] = date
    df.set_index('date', inplace=True)
    parameters = {'axes.labelsize': 18,
          'legend.fontsize': 12,
           'xtick.labelsize': 16,
            'ytick.labelsize':16}
    plt.rcParams.update(parameters)
    fig = plt.figure(figsize=figsize)
    for decile in df.columns[::-1]:
        s = df.loc[:,decile]
        s = s.cumsum()
        plt.plot(s.index, s, label = decile)
    if axvline:
        plt.axhline(y=0, color='gray', linestyle='--')
    plt.ylabel(ylabel)
    if legend:
        if legend_side:
            plt.legend(bbox_to_anchor=(1.01,0.5), loc="center left")            
        else:
            plt.legend(loc='lower left')
    plt.ylim(ylim)
    plt.tight_layout()
    if plotPath:
        plt.savefig(os.path.join(plotPath, name+'.png'))

def form_decile_portfolio_all(returns, chara_data, D, extreme_method = 0):
    T, N, L= chara_data.shape 
    portfolio_decile= np.zeros((T, L*D))
    portfolio_decile_time = np.zeros((T, L*D))
    length_output = 0
    for i in range(T):
        nonzero_index_alpha= np.argwhere((returns[i,:]!=-99.99))
        
        for j in range(L):
            nonzero_index_chara= np.argwhere(chara_data[i,:,j]!=-99.99)
            nonzero_inner = np.intersect1d(nonzero_index_alpha, nonzero_index_chara)
            chara_data_t= chara_data[i,nonzero_inner,j]
            index= np.argsort(chara_data_t)
            length= int(len(index)/D)
            for k in range(D):
                if k<D-1:
                    index_k= index[int(length*k):int(length*(k+1))]
                else:
                    if extreme_method == 0:
                        index_k= index[int(length*k):]
                    elif extreme_method == 1:
                        index_k= index[-length:]
                length_output+= len(list(index_k))
        
                if k==D-1:
                    chara_data_t[index_k]-=  np.min(chara_data_t[index_k])
                else:
                    chara_data_t[index_k]-= np.max(chara_data_t[index_k])                  
                   
                portfolio_decile[i, j*D+k]= np.mean(returns[i,list(nonzero_inner[index_k])])
                if np.sum(chara_data_t[index_k])==0:
                    portfolio_decile_time[i, j*D+k] = portfolio_decile[i, j*D+k]
                else:
                    portfolio_decile_time[i, j*D+k] = np.sum(returns[i,list(nonzero_inner[index_k])]*chara_data_t[index_k])/np.sum(chara_data_t[index_k])     
    return portfolio_decile, portfolio_decile_time

def plot_decile_portfolios(df, n_train=240, n_valid=60, dateEnd='20161201', plotPath=None, figsize=(8,6), name = '', ylabel = 'Cumulative expense ratio', axvline = True, segments = [], ylim = [-0.25, 1.8], legend = True, legend_side = True):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'tab:blue', 'tab:purple', 'tab:orange', 'tab:brown']    
    date = pd.date_range(end=dateEnd, periods=df.shape[0], freq='MS')
    df.loc[:,'date'] = date
    df.set_index('date', inplace=True)
    fig = plt.figure(figsize=figsize)
    for k, decile in enumerate(df.columns):
        s = df.loc[:,decile]
        if decile=='Sentiment' and axvline:
            s = pd.concat([s.iloc[:n_train],
                s.iloc[n_train:(n_train+n_valid)],
                s.iloc[(n_train+n_valid):]])
            plt.plot(s.index, s, label=decile)
        elif axvline:
            s = pd.concat([s.iloc[:n_train].cumsum() ,
                s.iloc[n_train:(n_train+n_valid)].cumsum(),
                s.iloc[(n_train+n_valid):].cumsum()])
            plt.plot(s.index, s, label=decile)            
        else:
            s = s.cumsum() 
            plt.plot(s.index, s, color = colors[k], label=decile)        
    if axvline:        
        plt.axvline(x=df.index[n_train], color='gray', linestyle='--')
        plt.axvline(x=df.index[n_train+n_valid], color='gray', linestyle='--')      
    plt.ylim(ylim)
    plt.ylabel(ylabel)  
    if legend:
        if legend_side:
            plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
        else:
            plt.legend()            
        plt.tight_layout()
    if plotPath:
        plt.savefig(os.path.join(plotPath, name+'.png'))


def cov_estimate(Y, lag):
    Y_mean = np.mean(Y)
    cov_estimate = 0
    T = Y.shape[0]
    for i in range(lag):
        if i>0:
            for t in range(i, T):
                cov_estimate+= 2*(1- i/(lag))*1/T*(Y[t] - Y_mean)*(Y[t-i]- Y_mean)
        else:
            for t in range(i, T):
                cov_estimate+= (1- i/(lag))*1/T*(Y[t] - Y_mean)*(Y[t-i]- Y_mean)            
    return np.sqrt(cov_estimate)

def significance(portfolio_decile, names, sentiment, start = 0, end = 200, D = 5, factor = True, sentiment_use = True, mean_std = True, output_more = False, adjust_mean = 1, adjust_t_sharpe = 1, use_std = False, use_robust = False, lag = 1):
    T, N  = portfolio_decile.shape
    if factor:
        N= N
    else:
        N = N//D
    output = []
    sharpe_output = []
    t_output = []
    
    for i in range(N):
        if factor:
            Y = portfolio_decile[start:end, i]
        else:
            Y = portfolio_decile[start:end,D*i+D-1] - portfolio_decile[start:end,D*i] 
            
        X = np.ones((T-start))
#         Y = Y/np.std(Y, axis = 0)
        Y = Y * 100
        mod = sm.OLS(Y,X)

        fii = mod.fit()
        if use_robust:
            fii = fii.get_robustcov_results(cov_type='HAC', maxlags=lag)
        
        mean = fii.summary2().tables[1]['Coef.']
        t_stats = fii.summary2().tables[1]['t']
        p_values = fii.summary2().tables[1]['P>|t|']
        std_err =   fii.summary2().tables[1]['Std.Err.']
        if use_robust:
#             std = cov_hac(Y, nlags = 12)
            std = cov_estimate(Y, lag = lag)
        else:
            std = np.std(Y)
        sharpe_ratio = (mean['const']/std)
        if factor:
            sharpe_output.append([sharpe_ratio, mean['const']*adjust_mean, std*adjust_t_sharpe, t_stats['const']])
        else:
            sharpe_output.append([sharpe_ratio, mean['const']*adjust_mean, std*adjust_t_sharpe, t_stats['const']])
            
        t_output.append(t_stats['const'])
        mean_direct = np.mean(Y)
        
        if sentiment_use:
            regression_sentiment = regression(Y, sentiment[start:]*100)
        else:
            regression_sentiment = ''
    
        name_print = '\_'.join(names[i].split('_'))
        
        if mean_std:
#             regression_mean = '&%0.2f'%(mean['const']*adjust_mean)+'&%0.2f'%(std*adjust_t_sharpe)
            regression_mean = '&%0.2f'%(mean['const']*adjust_mean)
#             regression_mean = '&%0.2f'%(mean_direct)+'&%0.2f'%(std*adjust_t_sharpe)
        else:
            regression_mean = ''
            
        sharpe_ratio = sharpe_ratio*adjust_mean/adjust_t_sharpe 
        if use_std:
#             regression_std = '&%0.2f'%(std_err['const'])
            regression_std = '&%0.2f'%(std*adjust_t_sharpe)
        else:
            regression_std = ''
        if p_values['const']<0.01:
#             output.append(name_print+regression_mean+regression_std+'&%0.2f'%(sharpe_ratio)+'&%0.1f'%(t_stats['const'])+'***'+ regression_sentiment )
            output.append(name_print+'&%0.2f'%(sharpe_ratio)+regression_mean+'***'+ regression_sentiment+regression_std )
#             +'\\\\'
        elif p_values['const']<0.05:
#             output.append(name_print+regression_mean+regression_std+'&%0.2f'%(sharpe_ratio)+'&%0.1f'%(t_stats['const'])+'**'+ regression_sentiment )
            output.append(name_print+'&%0.2f'%(sharpe_ratio)+regression_mean+'**'+ regression_sentiment+regression_std )
        elif p_values['const']<0.1:
#             output.append(name_print+regression_mean+regression_std+'&%0.2f'%(sharpe_ratio)+'&%0.1f'%(t_stats['const'])+'*'+ regression_sentiment )
            output.append(name_print+'&%0.2f'%(sharpe_ratio)+regression_mean+'**'+ regression_sentiment+regression_std )
        else:
#             output.append(name_print+regression_mean+regression_std+'&%0.2f'%(sharpe_ratio)+ '&%0.1f'%(t_stats['const'])+regression_sentiment)
            output.append(name_print+'&%0.2f'%(sharpe_ratio)+regression_mean+regression_sentiment+regression_std )
#         output.append(name_print+regression_mean+'&%0.3f'%(sharpe_ratio*adjust_t_sharpe)+ '&%0.3f'%(t_stats['const']*adjust_t_sharpe)+regression_sentiment+'\\\\')
    if output_more:
        return output, sharpe_output, t_output
    else:
        return output, sharpe_output
    
    
def return_name(x):
    if float(x[0])==0 and float(x[1])==59:
        output_string = 'Stock+ fund+ sentiment'
    if float(x[0])==46 and float(x[1])==59:
        output_string = 'Fund+ sentiment'
    if float(x[0])==59 and float(x[1])==45:
        output_string = 'Stock+ sentiment'
        
        
    if float(x[0])==0 and float(x[1])==58:
        output_string = 'Stock+ fund'
    if float(x[0])==0 and float(x[1])==45:
        output_string = 'Stock'
    if float(x[0])==46 and float(x[1])==58:
        output_string = 'Fund'

    if float(x[0])==59 and float(x[1])==53:
        output_string = 'Fund exclude momentum and flow'
    if float(x[0])==58 and float(x[1])==59:
        output_string = 'F_r12_2+ sentiment'
    if float(x[0])==56 and float(x[1])==47:
        output_string = 'Flow+ fund momentum+ sentiment'
    if float(x[0]) == 47 and float(x[1]) == 59:
        output_string = 'Flow+ F_r12_2+ sentiment'        
    return output_string 


def significance_helper(portfolio_decile):
    output = []
    sharpe_output = []
    Y = portfolio_decile
    X = np.ones((Y.shape[0]))
    mod = sm.OLS(Y,X)
    fii = mod.fit()
    mean = fii.summary2().tables[1]['Coef.']
    t_stats = fii.summary2().tables[1]['t']
    p_values = fii.summary2().tables[1]['P>|t|']
        
    if p_values['const']<0.01:
        output.append('&%0.1f'%(t_stats['const'])+'***')
    elif p_values['const']<0.05:
        output.append('&%0.1f'%(t_stats['const'])+'**')
    elif p_values['const']<0.1:
        output.append('&%0.1f'%(t_stats['const'])+'*')
    else:
        output.append( '&%0.1f'%(t_stats['const']))
    return output


def print_regression_many(mat_reg, mat_p, print_name, institutions, R2= [],  R2_print = False, std_errs = [], std_err_print = False, mean_factor = [], std_factor = [], p_factor = [], print_factor = False, orders = [], filename= ""):
#     print_name = ['1st', '2nd', '3rd', '4th', '5th']
    T, N = mat_reg.shape
    print_list_final = ['Anomalies']
    print_list_final += ['&'+k for k in institutions]
    print_list_final.append('\\\\\n')
    print_list_final = ''.join(print_list_final)
    filename.write(print_list_final)
    for i in orders:
        print_list = list(mat_reg[i,:])
        print_list_final = [print_name[i]]
        print_list_final += ['&%0.2f'%(k) for k in print_list]
        if print_factor:    
            temp_string = '&%0.2f'%(mean_factor[i])
            if p_factor[i]<0.01:
                temp_string+= '***'
            elif p_factor[i]<0.05:
                temp_string+='**'
            elif p_factor[i]<0.1:
                temp_string+='*'
            print_list_final += [temp_string]
        if R2_print:
            print_list_final  += ['&%0.2f'%(R2[i])]
        

        for k in range(N//2):
            if mat_p[i,k]<0.01:
                 print_list_final[2*k+1]+= '***'
            elif mat_p[i,k]<0.05:
                 print_list_final[2*k+1]+='**'
            elif mat_p[i,k]<0.1:
                print_list_final[2*k+1]+='*'
        
        print_list_final[-1]+='\\\\\n'   
        
        print_list_final = ''.join(print_list_final)
        filename.write(print_list_final)
        
        print_list_final = []
        if std_err_print:
            print_list = list(std_errs[i,:])
            print_list_final += ['&(%0.2f)&'%(k) for k in print_list]
            if print_factor:
                temp_string = '&(%0.2f)'%(std_factor[i])
                print_list_final += [temp_string]
            print_list_final.append('\\\\\n')
            print_list_final = ''.join(print_list_final)
#            print (print_list_final)
            filename.write(print_list_final)
            
def get_return_weight(Rhat1, R, mask, length_decile, normalized = True):
    T = mask.shape[0]
    lists = [] 
    weight = []    
    for t in range(T):
        nonzero_index = np.where(mask[t]!=0)[0]      
        rank1 = np.argsort(Rhat1[t][nonzero_index])     
        num_decile = len(nonzero_index)//length_decile
        weight_ori1 = np.zeros((len(nonzero_index))) 
        weight_pre1 = np.zeros((len(nonzero_index))) 
        for i in range(length_decile):        
            indices1 = rank1[i*num_decile:(i+1)*num_decile]   
            predicted_returns1 = Rhat1[t][nonzero_index[indices1]]      
            if i==length_decile-1:             
                predicted_returns1 -= np.min(predicted_returns1)                    
                weight_ori1[indices1] = 1/len(indices1) 
                if normalized:
                    weight_pre1[indices1] = predicted_returns1/np.sum(predicted_returns1)
                else:
                    weight_pre1[indices1] = predicted_returns1                 
            if i==0:
                predicted_returns1 -= np.max(predicted_returns1)                     
                weight_ori1[indices1] = - 1/len(indices1)
                if normalized:                
                     weight_pre1[indices1] = -1*predicted_returns1/np.sum(predicted_returns1)
                else:
                     weight_pre1[indices1] = -1*predicted_returns1                                    
        weight.append([weight_ori1,weight_pre1])                
    return weight, lists
