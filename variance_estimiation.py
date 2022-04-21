# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:36:04 2022

@author: Calum Strange
"""


import pandas as pd
import numpy as np
import scipy
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math
from functions import extract_data, linear_slope, calculate_n_online

severson = pd.read_csv('./Data/Severson_2019.csv')
attia = pd.read_csv('./Data/Attia_2020.csv')
attia_pred = pd.read_csv('./Data/Attia_predicted_capacity.csv')
baum = pd.read_csv('./Data/Baumhoefer_2014.csv')
dechent_1 = pd.read_csv('./Data/dechent_2017.csv')
dechent_2 = pd.read_csv('./Data/dechent_2020.csv')

# for dechent 2020 drop cell 22
dechent_2 = dechent_2[dechent_2['cellID']!=22]


attia_pred_df, n_cells_attia_pred, exp_time_attia_pred = extract_data(attia_pred)
sev_df, n_cells_sev, exp_time_sev = extract_data(severson)
attia_df, n_cells_attia, exp_time_attia = extract_data(attia)
baum_df, n_cells_baum, exp_time_baum = extract_data(baum)
dech_1_df, n_cells_dech_1, exp_time_dech_1 = extract_data(dechent_1)
dech_2_df, n_cells_dech_2, exp_time_dech_2 = extract_data(dechent_2)


print('-----------')
print('Attia pred')
print(calculate_n_online(attia_pred_df, n_cells_attia_pred,
                         exp_time_attia_pred, percs=np.arange(0.2, 1.2, 0.2)))
print('-----------\n')

print('Attia')
print(calculate_n_online(attia_df, n_cells_attia,
                         exp_time_attia, percs=np.arange(0.2, 1.2, 0.2)))
print('-----------\n')

print('Dechent 2017')
print(calculate_n_online(dech_1_df, n_cells_dech_1,
                         exp_time_dech_1, percs=np.arange(0.2, 1.2, 0.2)))
print('-----------\n')

print('Dechent 2020')
print(calculate_n_online(dech_2_df, n_cells_dech_2,
                         exp_time_dech_2, percs=np.arange(0.2, 1.2, 0.2)))
print('-----------\n')

print('Severson')
print(calculate_n_online(sev_df, n_cells_sev,
                         exp_time_sev, percs=np.arange(0.2, 1.2, 0.2)))
print('-----------\n')


print('Baum')
print(calculate_n_online(baum_df, n_cells_baum,
                         exp_time_baum, percs=np.arange(0.2, 1.2, 0.2)))
print('-----------\n')



