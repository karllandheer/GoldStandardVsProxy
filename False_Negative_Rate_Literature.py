# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 19:33:27 2024

@author: karl
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
from scipy.optimize import least_squares

np.random.seed(42)

N = 500000

corr_values = [0.58, 1-1E-10] #empirical value at 3T, and gold standard comparison, 1 minus eps for numerical reasons

true_fnr = 0.2 
alpha = 0.05
GABA_conc = 0.312 #from Short echo-time Magnetic Resonance Spectroscopy in ALS, simultaneous quantification of glutamate and GABA at 3 T
Npoints_case = 9 #from same paper
Npoints_control = 10 #from same paper


Delta_manuscript = 0.312*(0.8352941176470589-1) #this one from MEGA- paper Decreased motor cortex γ-aminobutyric acid in amyotrophic lateral sclerosis
rand_b_list = np.linspace(-0.2,0,71) #a range of bias introduced by the use of proxy variables


std_c = 0.06 #from literature
std_d = 0.02
std_a = std_c #assumptions because unmeausred, assume same as proxy variables
std_b = std_d


fig, ax = plt.subplots()
for corr in corr_values:
    fnr_cd_list = []
    for Delta in rand_b_list:
        p_values_cd = []

        for ii in range(N):
    
            a = (np.random.randn((Npoints_control))*std_a+GABA_conc)
            b = (np.random.randn((Npoints_case))*std_b+GABA_conc)+Delta
    
            mu_alpha = GABA_conc*(1-corr)/np.sqrt(1-corr**2)

            c =corr*a+np.sqrt(1-corr**2)*(np.random.randn((Npoints_control))*std_c+mu_alpha)
            d =corr*b+np.sqrt(1-corr**2)*(np.random.randn((Npoints_case))*std_d+mu_alpha)
             
            
            return_value = ttest_ind(c, d, equal_var=False) 
            p_values_cd.append(return_value.pvalue)
            
            
        #false negative is any case when the p value is greater than alpha
        # fnr_ab = [np.sum((np.array(x)>alpha))/len(x) for x in list_of_ab]
        fnr_cd = [np.sum((np.array(x)>alpha))/len(x) for x in [p_values_cd]][0]
        fnr_cd_list.append(fnr_cd)
          
    ax.plot(rand_b_list,fnr_cd_list)

    closest_index = np.argmin(np.abs(rand_b_list - Delta_manuscript))
    print(f'for the correlation {corr} the measured delta of {Delta_manuscript} the FNR is {fnr_cd_list[closest_index]} for corr = {corr}')
    
    
    fnr_cd_list = np.array(fnr_cd_list)
    index_fnr_below_0p05 = np.where(fnr_cd_list<0.05)[0][-1]
    print(f'for the correlation {corr} the smallest Delta required to get a FNR < 0.05 is {rand_b_list[index_fnr_below_0p05]} for corr = {corr}')
    

corr_values[-1] = 1 #replace 0.9999 with 1 just for display
legend_string = [f'$ρ = {x}$' for x in corr_values]
ax.legend(legend_string,fancybox=True)
ax.tick_params(which='both', top=True, right=True, labeltop=False, labelright=False) # Ticks, but no labels on top/right by default
fig.savefig('C:\ColumbiaWork\PapersOnGo\MEGAvsNon\MEGAvsNon\Figures\FNR_literature.png',dpi=300,bbox_inches='tight')
ax.set_ylabel('False negative rate (unitless)')
ax.set_xlabel(f'Difference between means of gold-standard measurements, $\Delta$ (unitless)')
#fig.savefig('C:\ColumbiaWork\PapersOnGo\MEGAvsNon\MEGAvsNon\Figures\FNR_literature.png',dpi=300,bbox_inches='tight')