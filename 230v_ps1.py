# -*- coding: utf-8 -*-
"""
Created on Wed Jul 06 15:36:20 2016

@author: MyPC
"""

import numpy as np
import pandas as pd
import credit_risk_lib as crm
import matplotlib.pyplot as plt

import os
print(os.getcwd() + "\n") 



cusip = "47035520"; #"24823Q10"
fundamental_file_name =  "Data/" + cusip + "_COMPUSTAT.csv"
market_file_name = "Data/" + cusip + "_CRSP.csv"

#%% Altman
df_altman = crm.load_altman_data_from_wrds(fundamental_file_name)

altman_z_score = crm.get_altman_z_score(df_altman)
# plot z-score on an inverted axis
plt.figure()
plt.plot(-altman_z_score)
plt.ylabel("Altman z-score (inverted axis)")
plt.title("Figure 1. Altman z-score 3 Years Prior to delisting (cusip=%s)" %(cusip))
plt.savefig("Out/Altman_z_score_%s.png" %(cusip))

#%% Ohlson-O score
df_ohlson =  crm.load_ohlson_o_score_data_from_wrds(fundamental_file_name)
ohlson_o_score = crm.get_ohlson_o_score(df_ohlson)
plt.figure()
plt.plot(ohlson_o_score)
plt.ylabel("Ohlson-O")
plt.title("Figure 2. Ohlson-O score 3 Years Prior to delisting (cusip=%s)" %(cusip))
plt.savefig("Out/Ohlson_o_score_%s.png" %(cusip))

#%% CHS
df_chs = crm.load_chs_data(fundamental_file_name = fundamental_file_name, 
                  security_market_file_name=market_file_name, index_market_file_name = "Data/SP_CRSP.csv")
                  
(score, prob_default)   = crm.get_prob_default_chs(df_chs,  predict_horizon=1)       
plt.figure()
plt.plot(np.log(prob_default))
plt.plot(np.log(df_chs.PRC))
plt.ylabel("CHS Logit model (log-scale)")
plt.title("Figure 3. CHS Logit mode 3 Years Prior to delisting (cusip=%s)" %(cusip))
plt.legend(["Predicted Default Prob","Equity Price"],loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.3))
plt.savefig("Out/CHS_%s.png" %(cusip))
    
'''
       NIMTA   TLMTA    EXRET     RSIZE   SIGMA  CASHMTA     MB   PRICE  
Mean   0.000   0.445   -0.011   -10.456   0.562    0.084  2.041   2.019  
Median 0.006   0.427   -0.009   -10.570   0.471    0.045  1.557   2.474
'''
df_chs[["NIMTAAVG","TLMTA","EXRETAVG","SIGMA","RSIZE","CASHMTA","MB","PRICE"]].describe()