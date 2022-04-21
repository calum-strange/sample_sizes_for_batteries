# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:15:21 2022

@author: Calum
"""

import pandas as pd
import numpy as np
import scipy
from sklearn.linear_model import LinearRegression

def extract_data(data, time='time'):
    # set time = 'cycle' to perform analysis of capacity against cycles
    n_cells = np.max(data['cellID'])
    exp_time = np.max(data[time])
    df = {}
    for i in range(1, n_cells+1):
        cell_data = data.loc[data['cellID'] == i]
        df[i] = np.array([cell_data[time], cell_data['capacity']])
    return df, n_cells, exp_time


def linear_slope(x,y):
    # fits a linear regression model to data and returns the slope of the fit
    reg = LinearRegression().fit(x, y)
    return reg.coef_[0]



def calculate_n_online(df, n_cells, experiment_time,
                       max_acceptable_dev=25, confidence_level=68,
                       percs=[1], n_boot_samples=1000):
    best_ns = {}
    
    for perc in percs:
        slopes = []
        for i in range(1, n_cells+1):
            x = np.array(df[i][0])
            y = np.array(df[i][1])

            y = y[x <= perc * experiment_time]
            x = x[x <= perc * experiment_time]

            x = x.reshape(-1, 1)
            
            if len(x) == 0:
                slopes.append(np.nan)
            else:
                slopes.append(linear_slope(x, y))


        sigma_hat = np.std(slopes)

        emp_RSEs = []
        best_n = n_cells

        for n in range(2, n_cells+1):
            boot_SEs = []
            for boot in range(n_boot_samples):
                random_cells = np.random.randint(0, high=n_cells, size=n)
                samples = []
                for cell in random_cells:
                    samples.append(slopes[cell])
                boot_SEs.append(np.std(samples))
            RSE = 100*np.std(boot_SEs)/sigma_hat
            
            emp_RSEs.append(RSE)
            
            q = - max_acceptable_dev/RSE
            prob = 100*(1-2*scipy.stats.norm.cdf(q))
            
            if prob >= confidence_level:
                if not best_ns.get(np.round(100*perc)):
                    best_ns[np.round(100*perc)] = n
                
    return best_ns



def calculate_RSEs_and_probs(df, n_cells,
                             max_acceptable_dev=25,
                             n_boot_samples = 1000):
    slopes = []
    for i in range(1, n_cells+1):
        x = np.array(df[i][0])
        y = np.array(df[i][1])

        x = x.reshape(-1, 1)
        
        if len(x) == 0:
            slopes.append(np.nan)
        else:
            slopes.append(linear_slope(x, y))


    theo_RSE = [100 / np.sqrt(2*n) for n in range(1, n_cells)]


    sigma_hat = np.std(slopes)

    n_boot_samples = 1000
    emp_RSEs = []        
    probs_emp = []
    probs_theo = []
    
    for n in range(2, n_cells+1):
        boot_SEs = []
        for boot in range(n_boot_samples):
            random_cells = np.random.randint(0, high=n_cells, size=n)
            samples = []
            for cell in random_cells:
                samples.append(slopes[cell])
            boot_SEs.append(np.std(samples))
            
        RSE = 100*np.std(boot_SEs)/sigma_hat
        
        emp_RSEs.append(RSE)

        ## calculating probabilities via standard normal
        q_emp = -  max_acceptable_dev/RSE
        probs_emp.append(100*(1-2*scipy.stats.norm.cdf(q_emp)))
        
        q_theo = -  max_acceptable_dev/theo_RSE[n-2]
        probs_theo.append(100*(1-2*scipy.stats.norm.cdf(q_theo)))


    return emp_RSEs, theo_RSE, probs_emp, probs_theo