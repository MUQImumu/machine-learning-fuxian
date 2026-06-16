import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D

def plotWeight3D(x, y, v, zs, xlabel, ylabel, zlabel, plotPath=None, idx_x=None, idx_y=None, idx_z=None, figsize=(8,6), label='weight'):
    parameters = {'axes.labelsize': 15,
          'legend.fontsize': 12,
           'xtick.labelsize': 11,
            'ytick.labelsize':11}
    plt.rcParams.update(parameters)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    levels = np.linspace(np.min(v),np.max(v), 51)
    for k in range(len(zs)):
        z = zs[k]
        im = ax.contourf(x, y, v[:,:,k], offset=z, levels=levels, cmap=cm.magma_r)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_zlim(zs[0], zs[-1])
    cbar = plt.colorbar(im, format='%.2f')
    cbar.ax.set_ylabel(label)
    plt.tight_layout()    
    if plotPath:
        plt.savefig(plotPath + 'Figure14.png')

def plotcontourmean_cross(dl, name_x, name_y, name_z, plotPath=None, sampleFreqPerAxis=50, figsize=(8,6), xlim=[-0.5,0.5], label='Abnormal return prediction', name = '', length = 14, cross_idx_num = 3, legend = True):    
    # First load the x and vs. Then take average. 
    v_list = []        
    for cross_idx in range(cross_idx_num):
        v = np.load('../deep_learning/result_saved/random_sampling/parsimonious/ave_mean_3d_'\
                    +str(cross_idx)+str(name_x)+str(name_y)+str(name_z)+'3.npy')       
        v_list.append(v)         
    v = np.array(v_list).mean(axis=0)
    x = np.linspace(xlim[0], xlim[1], sampleFreqPerAxis + 1)
    y = np.linspace(xlim[0], xlim[1], sampleFreqPerAxis + 1) 
    zs = [-0.35, -0.11,   0.15 ,  0.58 ,  0.922]  
    xlabel = 'flow'
    ylabel = 'F_r12_2'
    zlabel = 'sentiment'    
    plotWeight3D(x, y, v*100, zs, xlabel, ylabel, zlabel, plotPath=plotPath, idx_x=0, idx_y=1, idx_z=2, figsize=figsize, label=label)

plotcontourmean_cross(None, 'flow', 'F_r12_2', 'sentiment', plotPath = 'plots/', figsize = (10,8))    