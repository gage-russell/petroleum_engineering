# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 21:16:54 2018

@author: Owner
"""
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
data=pd.read_csv('C:/Users/Owner/Downloads/monthly-production.csv')
data=data[np.isfinite(data.Oil)]
maxidx=data[data.Oil==data.Oil.max()].index.values[0]
df=data.iloc[maxidx:]
df['month']=range(maxidx, len(data))
T=np.array(df.month)
Q=np.array(df.Oil)
Q=Q/1000
"""
"""
EXPONENTIAL
"""
def exp_decline(Q, T, T_end=None, q_f=None, q_i=None, plot=False):
    if T_end!=None:
        if q_f!=None:
            raise Exception("One or the other dumb fuck")
    if q_f!=None:
        if T_end!=None:
            raise Exception("One or the other dumb fuck")         
    q_i=Q[0] 
    def exponential_decline(q_i):
        def exp_dec(T,a):
            return q_i*np.exp(-a*T)
        return exp_dec
    exp_dec=exponential_decline(q_i)
    a_exp, a_exp_cov=curve_fit(exp_dec, T, Q, method="trf")
    pred_obs_exp=exp_dec(T, a_exp[0])
    if T_end!=None:
        T_to_end=range(T_end)
        total_pred_exp=exp_dec(T_to_end, a_exp[0])
        T_end_obs=max(T)
        T_future=range(T_end_obs, T_end)
        future_pred_exp=exp_dec(T_future, a_exp[0])
        if plot==True:
            fig, ax = plt.subplots(1, figsize=(16, 16))
            ax.set_title("Exponential Decline Curve Analysis", fontsize=30)
            label_size = 20
            yed = [tick.label.set_fontsize(label_size) for tick in ax.yaxis.get_major_ticks()]
            xed = [tick.label.set_fontsize(label_size) for tick in ax.xaxis.get_major_ticks()]
            ax.scatter(T, Q, color="black", marker="x", s=250, linewidth=3)
            ax.set_xlabel("Time (mos.)", fontsize=25)
            ax.set_ylabel("Oil Rate (1000 STB/d)", fontsize=25)
            ax.plot(T_to_end, total_pred_exp, color="red", linewidth=5, alpha=0.5, label="Exponential")
            ax.ticklabel_format(fontsize=25)
            ax.legend(fontsize=25)
            plt.show()
        return pred_obs_exp, total_pred_exp, future_pred_exp, a_exp, a_exp_cov
    elif q_f!=None:
        T_end_obs=max(T)
        T_iter=max(T)
        q_pred=q_f+1
        while q_pred>q_f:
            T_iter_range=range(T_iter+1)
            T_future_range=range(T_end_obs, T_iter+1)
            total_pred_exp=exp_dec(T_iter_range, a_exp[0])
            future_pred_exp=exp_dec(T_future_range, a_exp[0])
            q_pred=total_pred_exp[T_iter]
            T_iter+=1
            T_to_end=T_iter_range
        if plot==True:
            fig, ax = plt.subplots(1, figsize=(16, 16))
            ax.set_title("Exponential Decline Curve Analysis", fontsize=30)
            label_size = 20
            yed = [tick.label.set_fontsize(label_size) for tick in ax.yaxis.get_major_ticks()]
            xed = [tick.label.set_fontsize(label_size) for tick in ax.xaxis.get_major_ticks()]
            ax.scatter(T, Q, color="black", marker="x", s=250, linewidth=3)
            ax.set_xlabel("Time (mos.)", fontsize=25)
            ax.set_ylabel("Oil Rate (1000 STB/d)", fontsize=25)
            ax.plot(T_to_end, total_pred_exp, color="red", linewidth=5, alpha=0.5, label="Exponential")
            ax.ticklabel_format(fontsize=25)
            ax.legend(fontsize=25)
            plt.show()
        return pred_obs_exp, total_pred_exp, future_pred_exp, a_exp, a_exp_cov
    else:
        if plot==True:
                fig, ax = plt.subplots(1, figsize=(16, 16))
                ax.set_title("Exponential Decline Curve Analysis", fontsize=30)
                label_size = 20
                yed = [tick.label.set_fontsize(label_size) for tick in ax.yaxis.get_major_ticks()]
                xed = [tick.label.set_fontsize(label_size) for tick in ax.xaxis.get_major_ticks()]
                ax.scatter(T, Q, color="black", marker="x", s=250, linewidth=3)
                ax.set_xlabel("Time (mos.)", fontsize=25)
                ax.set_ylabel("Oil Rate (1000 STB/d)", fontsize=25)
                ax.plot(T, pred_obs_exp, color="red", linewidth=5, alpha=0.5, label="Exponential")
                ax.ticklabel_format(fontsize=25)
                ax.legend(fontsize=25)
                plt.show()
        return pred_obs_exp, a_exp, a_exp_cov


"""
HYPERBOLIC
"""

def hyp_decline(Q, T, T_end=None, q_f=None, q_i=None, plot=False):
    if T_end!=None:
        if q_f!=None:
            raise Exception("One or the other dumb fuck")
    if q_f!=None:
        if T_end!=None:
            raise Exception("One or the other dumb fuck")         
    q_i=Q[0]
    def hyperbolic_decline(q_i):
        def hyp_dec(T, a_i, b):
            return q_i/np.power((1+b*a_i*T), 1./b)
        return hyp_dec
    hyp_dec=hyperbolic_decline(q_i)
    a_hyp, a_hyp_cov=curve_fit(hyp_dec, T, Q, method="trf")
    pred_obs_hyp=hyp_dec(T, a_hyp[0], a_hyp[1])
    if T_end!=None:
        T_to_end=range(T_end)
        total_pred_hyp=hyp_dec(T_to_end, a_hyp[0], a_hyp[1])
        T_end_obs=max(T)
        T_future=range(T_end_obs, T_end)
        future_pred_hyp=hyp_dec(T_future, a_hyp[0], a_hyp[1])
        if plot==True:
            fig, ax = plt.subplots(1, figsize=(16, 16))
            ax.set_title("hyperbolic Decline Curve Analysis", fontsize=30)
            label_size = 20
            yed = [tick.label.set_fontsize(label_size) for tick in ax.yaxis.get_major_ticks()]
            xed = [tick.label.set_fontsize(label_size) for tick in ax.xaxis.get_major_ticks()]
            ax.scatter(T, Q, color="black", marker="x", s=250, linewidth=3)
            ax.set_xlabel("Time (mos.)", fontsize=25)
            ax.set_ylabel("Oil Rate (1000 STB/d)", fontsize=25)
            ax.plot(T_to_end, total_pred_hyp, color="red", linewidth=5, alpha=0.5, label="Hyperbolic")
            ax.ticklabel_format(fontsize=25)
            ax.legend(fontsize=25)
            plt.show()
        return pred_obs_hyp, total_pred_hyp, future_pred_hyp, a_hyp, a_hyp_cov
    elif q_f!=None:
        T_end_obs=max(T)
        T_iter=max(T)
        q_pred=q_f+1
        while q_pred>q_f:
            T_iter_range=range(T_iter+1)
            T_future_range=range(T_end_obs, T_iter+1)
            total_pred_hyp=hyp_dec(T_iter_range, a_hyp[0], a_hyp[1])
            future_pred_hyp=hyp_dec(T_future_range, a_hyp[0], a_hyp[1])
            q_pred=total_pred_hyp[T_iter]
            T_iter+=1
            T_to_end=T_iter_range
        if plot==True:
            fig, ax = plt.subplots(1, figsize=(16, 16))
            ax.set_title("Hyperbolic Decline Curve Analysis", fontsize=30)
            label_size = 20
            yed = [tick.label.set_fontsize(label_size) for tick in ax.yaxis.get_major_ticks()]
            xed = [tick.label.set_fontsize(label_size) for tick in ax.xaxis.get_major_ticks()]
            ax.scatter(T, Q, color="black", marker="x", s=250, linewidth=3)
            ax.set_xlabel("Time (mos.)", fontsize=25)
            ax.set_ylabel("Oil Rate (1000 STB/d)", fontsize=25)
            ax.plot(T_to_end, total_pred_hyp, color="red", linewidth=5, alpha=0.5, label="Hyperbolic")
            ax.ticklabel_format(fontsize=25)
            ax.legend(fontsize=25)
            plt.show()
        return pred_obs_hyp, total_pred_hyp, future_pred_hyp, a_hyp, a_hyp_cov
    else:
        if plot==True:
                fig, ax = plt.subplots(1, figsize=(16, 16))
                ax.set_title("Hyperbolic Decline Curve Analysis", fontsize=30)
                label_size = 20
                yed = [tick.label.set_fontsize(label_size) for tick in ax.yaxis.get_major_ticks()]
                xed = [tick.label.set_fontsize(label_size) for tick in ax.xaxis.get_major_ticks()]
                ax.scatter(T, Q, color="black", marker="x", s=250, linewidth=3)
                ax.set_xlabel("Time (mos.)", fontsize=25)
                ax.set_ylabel("Oil Rate (1000 STB/d)", fontsize=25)
                ax.plot(T, pred_obs_hyp, color="red", linewidth=5, alpha=0.5, label="Hyperbolic")
                ax.ticklabel_format(fontsize=25)
                ax.legend(fontsize=25)
                plt.show()
        return pred_obs_hyp, a_hyp, a_hyp_cov
"""
PARABOLIC
"""
def parab_decline(Q, T, T_end=None, q_f=None, q_i=None, plot=False):
    if T_end!=None:
        if q_f!=None:
            raise Exception("One or the other dumb fuck")
    if q_f!=None:
        if T_end!=None:
            raise Exception("One or the other dumb fuck")         
    q_i=Q[0]
    def parabolic_decline(q_i):
        def parab_dec(T, a_i):
            return q_i/(1+a_i*T)
        return parab_dec
    parab_dec=parabolic_decline(q_i)
    a_parab, a_parab_cov=curve_fit(parab_dec, T, Q, method="trf")
    pred_obs_parab=parab_dec(T, a_parab[0])
    if T_end!=None:
        T_to_end=range(T_end)
        total_pred_parab=parab_dec(T_to_end, a_parab[0])
        T_end_obs=max(T)
        T_future=range(T_end_obs, T_end)
        future_pred_parab=parab_dec(T_future, a_parab[0])
        if plot==True:
            fig, ax = plt.subplots(1, figsize=(16, 16))
            ax.set_title("Parabolic Decline Curve Analysis", fontsize=30)
            label_size = 20
            yed = [tick.label.set_fontsize(label_size) for tick in ax.yaxis.get_major_ticks()]
            xed = [tick.label.set_fontsize(label_size) for tick in ax.xaxis.get_major_ticks()]
            ax.scatter(T, Q, color="black", marker="x", s=250, linewidth=3)
            ax.set_xlabel("Time (mos.)", fontsize=25)
            ax.set_ylabel("Oil Rate (1000 STB/d)", fontsize=25)
            ax.plot(T_to_end, total_pred_parab, color="red", linewidth=5, alpha=0.5, label="Parabolic")
            ax.ticklabel_format(fontsize=25)
            ax.legend(fontsize=25)
            plt.show()
        return pred_obs_parab, total_pred_parab, future_pred_parab, a_parab, a_parab_cov
    elif q_f!=None:
        T_end_obs=max(T)
        T_iter=max(T)
        q_pred=q_f+1
        while q_pred>q_f:
            T_iter_range=range(T_iter+1)
            T_future_range=range(T_end_obs, T_iter+1)
            total_pred_parab=parab_dec(T_iter_range, a_parab[0])
            future_pred_parab=parab_dec(T_future_range, a_parab[0])
            q_pred=total_pred_parab[T_iter]
            T_iter+=1
            T_to_end=T_iter_range
        if plot==True:
            fig, ax = plt.subplots(1, figsize=(16, 16))
            ax.set_title("Parabolic Decline Curve Analysis", fontsize=30)
            label_size = 20
            yed = [tick.label.set_fontsize(label_size) for tick in ax.yaxis.get_major_ticks()]
            xed = [tick.label.set_fontsize(label_size) for tick in ax.xaxis.get_major_ticks()]
            ax.scatter(T, Q, color="black", marker="x", s=250, linewidth=3)
            ax.set_xlabel("Time (mos.)", fontsize=25)
            ax.set_ylabel("Oil Rate (1000 STB/d)", fontsize=25)
            ax.plot(T_to_end, total_pred_parab, color="red", linewidth=5, alpha=0.5, label="Parabolic")
            ax.ticklabel_format(fontsize=25)
            ax.legend(fontsize=25)
            plt.show()
        return pred_obs_parab, total_pred_parab, future_pred_parab, a_parab, a_parab_cov
    else:
        if plot==True:
                fig, ax = plt.subplots(1, figsize=(16, 16))
                ax.set_title("Parabolic Decline Curve Analysis", fontsize=30)
                label_size = 20
                yed = [tick.label.set_fontsize(label_size) for tick in ax.yaxis.get_major_ticks()]
                xed = [tick.label.set_fontsize(label_size) for tick in ax.xaxis.get_major_ticks()]
                ax.scatter(T, Q, color="black", marker="x", s=250, linewidth=3)
                ax.set_xlabel("Time (mos.)", fontsize=25)
                ax.set_ylabel("Oil Rate (1000 STB/d)", fontsize=25)
                ax.plot(T, pred_obs_parab, color="red", linewidth=5, alpha=0.5, label="Parabolic")
                ax.ticklabel_format(fontsize=25)
                ax.legend(fontsize=25)
                plt.show()
        return pred_obs_parab, a_parab, a_parab_cov


"""
EXAMPLES
"""
"""
#Observed only decline curves
exp_prediction, a_exp, a_exp_covariance=exp_decline(Q, T)  #this is an example for exponential decline
hyp_prediction, a_hyp, a_hyp_covariance=hyp_decline(Q, T) #this is an example for hyperbolic decline
parab_prediction, a_parab, a_parab_covariance=parab_decline(Q, T) #this is an example of parabolic decline

#End time given decline curves
exp_prediction, total_pred_exp, future_pred_exp, a_exp, a_exp_covariance= exp_decline(Q, T, T_end=70)
hyp_prediction, total_pred_hyp, future_pred_hyp, a_hyp, a_hyp_covariance= hyp_decline(Q, T, T_end=70)
parab_prediction, total_pred_parab, future_pred_parab, a_parab, a_parab_covariance= parab_decline(Q, T, T_end=55)

#Economic flow limit given decline curves
exp_prediction, total_pred_exp, future_pred_exp, a_exp, a_exp_covariance= exp_decline(Q, T, q_f=0.5)
hyp_prediction, total_pred_hyp, future_pred_hyp, a_hyp, a_hyp_covariance= hyp_decline(Q, T, q_f=1)
parab_prediction, total_pred_parab, future_pred_parab, a_parab, a_parab_covariance= parab_decline(Q, T, q_f=15/1000)

#Failed due to T_end and q_f being defined
exp_prediction, a_exp, a_exp_covariance=exp_decline(Q, T, T_end=70, q_f=1) #this is an example of parabolic decline
hyp_prediction, a_hyp, a_hyp_covariance=hyp_decline(Q, T, T_end=70, q_f=1) #this is an example of parabolic decline
parab_prediction, a_parab, a_parab_covariance=parab_decline(Q, T, T_end=70, q_f=1) #this is an example of parabolic decline

#Exp with plot
exp_prediction, a_exp, a_exp_covariance=exp_decline(Q, T, plot=True)  #this is an example for exponential decline
a,b,c,d,e=exp_decline(Q, T, T_end=70, plot=True)  #this is an example for exponential decline
a,b,c,d,e=exp_decline(Q, T, q_f=0.6, plot=True)

#Hyp with plot
a,b,c=hyp_decline(Q, T, plot=True)  #this is an example for exponential decline
a,b,c,d,e=hyp_decline(Q, T, T_end=70, plot=True)  #this is an example for exponential decline
a,b,c,d,e=hyp_decline(Q, T, q_f=0.6, plot=True)

#Parab with plot
a,b,c=parab_decline(Q, T, plot=True)  #this is an example for exponential decline
a,b,c,d,e=parab_decline(Q, T, T_end=70, plot=True)  #this is an example for exponential decline
a,b,c,d,e=parab_decline(Q, T, q_f=0.6, plot=True)
"""
