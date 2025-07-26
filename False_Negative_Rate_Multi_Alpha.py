# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 19:33:27 2024

@author: karl
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

np.random.seed(42)

# N = 100000
N = 500000

N_corr_points = 31 #number of correlation points in the plot

corr_values = np.linspace(0,1,N_corr_points)
corr_values[corr_values==1] = 1 - 1E-10 #avoid exactly 1 due to 0/0

mu_A = 2.1 #the mean of the true control distribution

std_scale_factor = 1

Npoints_list = [10, 25, 100] #number of points sampled from distribution
alpha_list = [0.05, 0.01, 0.001] #false positive rate

Nx=len(Npoints_list)
Ny=len(alpha_list)

#array of values for big Delta
Delta_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
Delta_list = [x*2 for x in Delta_list]

#some plotting stuff
fig, axes = plt.subplots(
    nrows=Nx,
    ncols=Ny,
    figsize=(Nx * 5, Ny * 4),  # Adjust these multipliers as needed for plot readability
)

if Nx == 1 and Ny == 1:
    axes = np.array([[axes]])
elif Nx == 1:
    axes = axes[np.newaxis, :] # Make it 2D with one row
elif Ny == 1:
    axes = axes[:, np.newaxis] # Make it 2D with one column

for ii, Npoints in enumerate(Npoints_list):
    for jj, alpha in enumerate(alpha_list):
    
        list_of_ab = []
        list_of_cd = []
        
        for Delta in Delta_list:
            list_of_ab = []
            list_of_cd = []
            # fpr_list = []
            for corr in corr_values:
                p_values_ab = []
                p_values_cd = []
                for tt in range(N):
                    a = (np.random.randn((Npoints))*std_scale_factor+mu_A)
                    b = (np.random.randn((Npoints))*std_scale_factor+mu_A)+Delta
    
                    mu_alpha = mu_A*(1-corr)/np.sqrt(1-corr**2)
                    
                    c =corr*a+np.sqrt(1-corr**2)*(np.random.randn((Npoints))*std_scale_factor+mu_alpha)
                    d =corr*b+np.sqrt(1-corr**2)*(np.random.randn((Npoints))*std_scale_factor+mu_alpha)
                    
                    return_value = ttest_ind(c, d, equal_var=False) 
                    p_value = return_value.pvalue
                    p_values_cd.append(p_value)
                    
                # list_of_ab.append(p_values_ab)
                list_of_cd.append(p_values_cd)
                    
                #false negative is any case when the p value is greater than alpha
                # fnr_ab = [np.sum((np.array(x)>alpha))/len(x) for x in list_of_ab]
                fnr_cd = [np.sum((np.array(x)>alpha))/len(x) for x in list_of_cd]


            ax = axes[ii, jj]
            ax.plot(corr_values, fnr_cd)
    
            ax.grid(True, linestyle=':', alpha=0.6) # Add a grid
            
            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.02])


        if (ii == Nx - 1) and (jj == 1): # If it's the last row
            ax.set_xlabel(r'Correlation between proxy and gold-standard measurements, $\rho$ (unitless)', fontsize=12)

        if (jj == 0) and (ii == 1): # If it's the first column
            ax.set_ylabel('False negative rate (unitless)', fontsize=12)
            
        ax.set_title(f'$N_{{{{samples}}}} = {Npoints}$, $\\alpha = {alpha:.3f}$')

legend_string = [f'$\Delta = {x}$ mM' for x in Delta_list]
plt.legend(legend_string,loc = 'center left', bbox_to_anchor=(1.02, 1.7),fancybox=True)
#plt.savefig('C:\ColumbiaWork\PapersOnGo\MEGAvsNon\MEGAvsNon\Figures\FalseNegativeRate.png',dpi=300,bbox_inches='tight')