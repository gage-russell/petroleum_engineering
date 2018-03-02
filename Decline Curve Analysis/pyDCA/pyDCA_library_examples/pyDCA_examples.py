# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 21:16:54 2018

@author: GAGE RUSSELL
COPYRIGHT gagetyrussell
"""

from pyDCA import *
"""
#####################################################################################################
SINGLE WELL DATA FORMAT EXAMPLE (SINGLE WELL)
#####################################################################################################
"""
path='C:/Users/Owner/Downloads/monthly-production.csv'
df, Q, T=single_well_ARPS_data_format(path, Oil=True, Gas=False, start_from_max=True)

"""
#####################################################################################################
ARPS DECLINE EXAMPLES (SINGLE WELL)
#####################################################################################################
"""
#Observed only decline curves
exp_prediction, a_exp, a_exp_covariance=exp_decline(Q, T)  #this is an example for exponential decline
hyp_prediction, a_hyp, a_hyp_covariance=hyp_decline(Q, T) #this is an example for hyperbolic decline
har_prediction, a_har, a_har_covariance=har_decline(Q, T) #this is an example of harmonic decline

#End time given decline curves
exp_prediction, total_pred_exp, future_pred_exp, a_exp, a_exp_covariance= exp_decline(Q, T, T_end=70)
hyp_prediction, total_pred_hyp, future_pred_hyp, a_hyp, a_hyp_covariance= hyp_decline(Q, T, T_end=70)
har_prediction, total_pred_har, future_pred_har, a_har, a_har_covariance= har_decline(Q, T, T_end=55)

#Economic flow limit given decline curves
exp_prediction, total_pred_exp, future_pred_exp, a_exp, a_exp_covariance= exp_decline(Q, T, q_f=15)
hyp_prediction, total_pred_hyp, future_pred_hyp, a_hyp, a_hyp_covariance= hyp_decline(Q, T, q_f=15)
har_prediction, total_pred_har, future_pred_har, a_har, a_har_covariance= har_decline(Q, T, q_f=15)

#Failed due to T_end and q_f being defined
exp_prediction, a_exp, a_exp_covariance=exp_decline(Q, T, T_end=70, q_f=15) #this is an example of harmonic decline
hyp_prediction, a_hyp, a_hyp_covariance=hyp_decline(Q, T, T_end=70, q_f=15) #this is an example of harmonic decline
har_prediction, a_har, a_har_covariance=har_decline(Q, T, T_end=70, q_f=15) #this is an example of harmonic decline

#Exp with plot
exp_prediction, a_exp, a_exp_covariance=exp_decline(Q, T, plot=True)  #this is an example for exponential decline
a,b,c,d,e=exp_decline(Q, T, T_end=70, plot=True)  #this is an example for exponential decline
a,b,c,d,e=exp_decline(Q, T, q_f=15, plot=True)

#Hyp with plot
a,b,c=hyp_decline(Q, T, plot=True)  #this is an example for exponential decline
a,b,c,d,e=hyp_decline(Q, T, T_end=70, plot=True)  #this is an example for exponential decline
a,b,c,d,e=hyp_decline(Q, T, q_f=15, plot=True)

#Har with plot
a,b,c=har_decline(Q, T, plot=True)  #this is an example for exponential decline
a,b,c,d,e=har_decline(Q, T, T_end=70, plot=True)  #this is an example for exponential decline
a,b,c,d,e=har_decline(Q, T, q_f=15, plot=True)

"""
#####################################################################################################
MULTI WELL DATA FORMAT EXAMPLE
#####################################################################################################
"""
path='C:/Users/Owner/Downloads/2wellsGrady Production Time Series.csv'
loopdf, Q, T=multi_well_ARPS_data_format(path, Oil=True, Gas=False)

"""
#####################################################################################################
ARPS DECLINE EXAMPLES (MULTI-WELL)
#####################################################################################################
"""
#Obeserved only decline curves
pred, a, cov=multi_exp_decline(Q, T)
pred, a, cov=multi_hyp_decline(Q, T)
pred, a, cov=multi_har_decline(Q, T)

#Decline curves with final time given
pred, total, future, a, cov=multi_exp_decline(Q, T, T_end=120)
pred, total, future, a, cov=multi_hyp_decline(Q, T, T_end=120)
pred, total, future, a, cov=multi_har_decline(Q, T, T_end=120)

#Decline curves with final time given
pred, total, future, a, cov=multi_exp_decline(Q, T, q_f=65)
pred, total, future, a, cov=multi_hyp_decline(Q, T, q_f=65)
pred, total, future, a, cov=multi_har_decline(Q, T, q_f=65)

#Obeserved only decline curves WITH PLOTS
pred, a, cov=multi_exp_decline(Q, T, plot=True)
pred, a, cov=multi_hyp_decline(Q, T, plot=True)
pred, a, cov=multi_har_decline(Q, T, plot=True)

"""
#####################################################################################################
EXAMPLE OF DATA FORMAT FOR FBPROPHET TIME SERIES FORECASTING DECLINE CURVE
#####################################################################################################
"""
#MULTIPLE WELLS (multi_well=True automatically)
path='C:/Users/Owner/Downloads/2wellsGrady Production Time Series.csv'
df=fbprophet_DCA_data_format(path)

#SINGLE WELL (must set multi_well=False)
singlepath='C:/Users/Owner/Downloads/monthly-production.csv'
df=fbprophet_DCA_data_format(singlepath, multi_wells=False)
"""
#####################################################################################################
EXAMPLES OF FBPROPHET TIME SERIES FORECASTING DECLINE CURVE
#####################################################################################################
"""
#SINGLE WELL (must specify multi_wells=False) *API number will just be set to 1 due to the way single well
#production data is downloaded from Drilling Info
eur, forecast_outputs, total_prediction=fbprophet_DCA(df, multi_wells=False)

#MULTIPLE WELLS
eur, forecast_outputs, total_prediction=fbprophet_DCA(df)

