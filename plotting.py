# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:36:04 2022

@author: Calum Strange
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
from functions import extract_data, linear_slope, calculate_RSEs_and_probs

severson = pd.read_csv('./Data/Severson_2019.csv')
attia = pd.read_csv('./Data/Attia_2020.csv')
attia_pred = pd.read_csv('./Data/Attia_predicted_capacity.csv')
baum = pd.read_csv('./Data/Baumhoefer_2014.csv')
dechent_1 = pd.read_csv('./Data/dechent_2017.csv')
dechent_2 = pd.read_csv('./Data/dechent_2020.csv')

## There is one anomalous cell for the Dechent 2020 dataset so we remove this
dechent_2 = dechent_2[dechent_2['cellID']!=22]


# extracting all of the required data for our plotting
sev_df, n_cells_sev, exp_time_sev = extract_data(severson)
attia_df, n_cells_attia, exp_time_attia = extract_data(attia)
attia_pred_df, n_cells_attia_pred, exp_time_attia_pred = extract_data(attia_pred)
baum_df, n_cells_baum, exp_time_baum = extract_data(baum)
dech_1_df, n_cells_dech_1, exp_time_dech_1 = extract_data(dechent_1)
dech_2_df, n_cells_dech_2, exp_time_dech_2 = extract_data(dechent_2)

# collecting data for plotting
dfs = [sev_df,  baum_df, attia_df, dech_1_df, dech_2_df, attia_pred_df]

n_cells = [n_cells_sev,  n_cells_baum, n_cells_attia,
           n_cells_dech_1, n_cells_dech_2, n_cells_attia_pred]

## Titles for plots
titles = ['Severson-2019', 'Baumhofer-2014', 'Attia-2020',
          'Dechent-2017', 'Dechent-2020', 'Attia-predicted']





### Plot of capacity curves
fig, axs = plt.subplots(2,3, figsize=(10,5))

j = 0
for i, df in enumerate(dfs):
    axs[j,i%3].set_title(titles[i])
    axs[j,i%3].set_xlabel('time (days)')
    axs[j,i%3].set_ylabel('SOH')
    for cell in df.keys():
        axs[j, i%3].plot(df[cell][0], df[cell][1], c='k', lw=0.5)
    if i == 2:
        j+=1
        
        
fig.tight_layout()
plt.savefig('./plots/data_against_time.pdf')
plt.show()


### QQ plots at 100%
fig, axs = plt.subplots(2,3, figsize=(10,7), sharey=True, sharex=True)

j = 0
for i, df in enumerate(dfs):
    axs[j,i%3].set_title(titles[i], size=14)
    axs[j,i%3].set_xlabel('Theoretical quantiles', size=12)
    axs[j,i%3].set_ylabel('Sample quantiles,', size=12)
    
    slopes = []
    for cell in range(1, n_cells[i]+1):
        x = np.array(df[cell][0])
        y = np.array(df[cell][1])

        x = x.reshape(-1, 1)
        
        if len(x) == 0:
            slopes.append(np.nan)
        else:
            slopes.append(linear_slope(x, y))
                
    sm.qqplot((np.array(slopes)-np.mean(slopes))/np.std(slopes), line ='q', ax=axs[j, i%3],
              c='k', markerfacecolor='k')
    
    if len(x)>1:
        axins1 = axs[j,i%3].inset_axes([0.45, 0.1, 0.5, 0.25])
        axins1.hist(np.array(slopes),20, color='k')
        
    if i == 2:
        j+=1
  
fig.tight_layout()
plt.savefig('./plots/Q-Q_plots.pdf')
plt.show()


    
### Plotting RSEs
fig, axs = plt.subplots(2,3, figsize=(10,5))

j = 0
for i, df in enumerate(dfs):
    axs[j,i%3].set_title(titles[i])
    axs[j,i%3].set_xlabel('Sample size')
    axs[j,i%3].set_ylabel('RSE (%)')
    axs[j,i%3].set_xlim(-5, np.max(n_cells) + 5)
    axs[j,i%3].set_ylim(-5, 80)
    emp_RSE, theo_RSE, _, _ = calculate_RSEs_and_probs(df, n_cells[i])
    axs[j, i%3].plot(theo_RSE, c='k', lw=0.5)
    axs[j, i%3].scatter(np.arange(len(emp_RSE)), emp_RSE, c='white', edgecolor='k')
    if i == 2:
        j+=1      
        
fig.tight_layout()
plt.savefig('./plots/RSEs_emp_and_theo.pdf')
plt.show()


### Plotting Probabilities

fig, axs = plt.subplots(2,3, figsize=(10,5))

j = 0
for i, df in enumerate(dfs):
    axs[j,i%3].set_title(titles[i])
    axs[j,i%3].set_xlabel('Sample size')
    axs[j,i%3].set_ylabel('Probability (%)')
    axs[j,i%3].set_xlim(-5, np.max(n_cells) + 5)
    axs[j,i%3].set_ylim(-5, 105)
    _, _, probs_emp, probs_theo = calculate_RSEs_and_probs(df, n_cells[i])
    axs[j, i%3].plot(probs_theo, c='k', lw=0.5)
    axs[j, i%3].scatter(np.arange(len(probs_emp)), probs_emp, c='white', edgecolor='k')
    if i == 2:
        j+=1      
        
fig.tight_layout()
plt.savefig('./plots/probs_emp_and_theo.pdf')
plt.show()



### QQ plots with percentage of input

## here we don't plot the Attia predicted datset

dfs = [sev_df,  baum_df, attia_df, dech_1_df, dech_2_df]
n_cells = [n_cells_sev,  n_cells_baum, n_cells_attia,
           n_cells_dech_1, n_cells_dech_2]

exp_times = [exp_time_sev,  exp_time_baum, exp_time_attia,
           exp_time_dech_1, exp_time_dech_2]

fig, axs = plt.subplots(5,5, figsize=(15,15), sharex=True, sharey=True)

j=0

for perc in np.arange(0.2, 1.2, 0.2):   
    for i, df in enumerate(dfs):
        axs[j,i].set_xlabel('')
        axs[j,i].set_ylabel('rty')
        slopes = []
        for cell in range(1, n_cells[i]+1):
            x = np.array(df[cell][0])
            y = np.array(df[cell][1])
            
            y = y[x <= perc * exp_times[i]]
            x = x[x <= perc * exp_times[i]]
    
            x = x.reshape(-1, 1)
            
            if len(x) == 0:
                slopes.append(np.nan)
            else:
                slopes.append(linear_slope(x, y))
                    
        sm.qqplot((np.array(slopes)-np.mean(slopes))/np.std(slopes), line ='q', ax=axs[j, i],
                  c='k', markerfacecolor='k', xlabel='', ylabel='')
        if len(x)>1:
            axins1 = axs[j,i].inset_axes([0.4, 0.1, 0.55, 0.3])
            axins1.hist(np.array(slopes),20, color='k')
           
            if i == 0:
                axins1.set_xlim([-0.8, 0])

            elif i == 1:
                axins1.set_xlim([-0.8, -0.15])

                axins1.set_ylim([0, 10])
            elif i == 2:
                axins1.set_xlim([-0.8, 0])

            elif i == 3 :
                axins1.set_xlim([-0.2, -0.152])

            else:
                axins1.set_xlim([-0.05, -0.015])


    j+=1
    for k in range(5):
        axs[4,k].set_xlabel('Theoretical Quantiles', size=15)
        axs[k,0].set_ylabel('Sample Quantiles', size=15)
        axs[0,k].set_title(titles[k], size=20, pad=20)        
        
axs[0,0].annotate('20%', (-80,80), xycoords='axes points', size=20, rotation='vertical')
axs[1,0].annotate('40%', (-80,80), xycoords='axes points', size=20, rotation='vertical')
axs[2,0].annotate('60%', (-80,80), xycoords='axes points', size=20, rotation='vertical')
axs[3,0].annotate('80%', (-80,80), xycoords='axes points', size=20, rotation='vertical')
axs[4,0].annotate('100%', (-80,80), xycoords='axes points', size=20, rotation='vertical')
            
fig.tight_layout()
plt.savefig('./plots/Q-Q_plots_different_percs.pdf')
plt.show()

