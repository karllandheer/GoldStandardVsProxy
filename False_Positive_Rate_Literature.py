# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 19:33:27 2024

@author: karl
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind, ttest_rel
from scipy.stats import gaussian_kde


np.random.seed(42)


N = 500000

corr = 0.72 #empirical value at 7T
rand_d_list = np.linspace(0,3,71) #a range of bias introduced by the use of proxy variables

alpha = 0.05

# pre_test_vals = [7.525371838,5.680068021,6.820934551,5.298153342,4.654821886,3.201262878,5.582929746,8.403816547,5.660714085,8.396797701,
# 9.357454303,6.597390981,7.039939054,6.826232273,5.999884486,5.795365613,8.551641439,6.335005035,7.032780391,9.430504797,9.689421735,
# 6.850126077,5.658890493,5.310078348,7.245422651,]

# post_test_vals = [6.518826216,8.17308868,6.479650693,7.978594028,6.53853625,6.159600158,4.169059855,10.83229785,6.097715304,
# 5.243369388,9.28770098,5.202086933,8.283323111,10.56449597,5.865212537,10.06930747,10.48830988,5.720077556,7.018505785,
# 10.91643984,8.579596291,5.809928574,5.120225956,8.148282808,10.37972639]

mean_pre_test_vals = 6.75780033084
mean_post_test_vals =  7.58575834012
Npoints = 25 #values from manuscript

GABA_conc = mean_pre_test_vals


delta_manuscript = (mean_post_test_vals-mean_pre_test_vals)/np.sqrt(1-corr**2)

#values from manuscript
std_scale_c = 1.550032652662542
std_scale_d = 2.0458394840559833

#assumed, because gold-standard measurements are obviously unknown
std_scale_a = std_scale_c
std_scale_b = std_scale_d


plt.figure()
ax = plt.subplot(111)
fpr_ab_list = []
fpr_cd_list = []


mu_alpha = GABA_conc*(1-corr)/np.sqrt(1-corr**2)
for delta in rand_d_list:
    p_values_cd = []
    
    for ii in range(N):
        
        #true signals, no relationship
        a = (np.random.randn((Npoints))*std_scale_a+GABA_conc)
        b = (np.random.randn((Npoints))*std_scale_b+GABA_conc)
        
        c =corr*a+np.sqrt(1-corr**2)*(np.random.randn((Npoints))*std_scale_c+mu_alpha)
        d =corr*b+np.sqrt(1-corr**2)*(np.random.randn((Npoints))*std_scale_d+mu_alpha+delta)

        
        return_value = ttest_ind(c, d, equal_var=False) 
        # return_value = ttest_rel(c, d)
        p_values_cd.append(return_value.pvalue)
        

    fpr_cd = [np.sum((np.array(x)<alpha))/len(x) for x in [p_values_cd]][0]
    fpr_cd_list.append(fpr_cd)


fig, ax = plt.subplots()
ax.plot(rand_d_list, fpr_cd_list)
ax.tick_params(which='both', top=True, right=True, labeltop=False, labelright=False) # Ticks, but no labels on top/right by default
ax.set_ylim([0, 1])
ax.set_ylabel('False positive rate (unitless)')
ax.set_xlabel(f'Bias introduced by use of proxy measurement, $\delta$ (unitless)')
#fig.savefig('C:\ColumbiaWork\PapersOnGo\MEGAvsNon\MEGAvsNon\Figures\FPR_literature.png',dpi=300,bbox_inches='tight')

closest_index = np.argmin(np.abs(rand_d_list - delta_manuscript))
print(f'for the measured delta of {delta_manuscript} the FPR is {fpr_cd_list[closest_index]}')

