# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 21:16:54 2018

@author: GAGE RUSSELL
COPYRIGHT gagetyrussell
"""
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fbprophet import Prophet

"""
#####################################################################################################
SINGLE WELL DATA FORMATING FUNCTION FOR ARPS (SINGLE WELL)
#####################################################################################################
"""
def single_well_ARPS_data_format(data_path, Oil=True, Gas=False, start_from_max=False):
    if start_from_max==True:
        if Oil==True:
            ddt=pd.read_csv(data_path)
            ddt=ddt[np.isfinite(ddt.Oil)]
            maxidx=ddt[ddt.Oil==ddt.Oil.max()].index.values[0]
            df=ddt.iloc[maxidx:]
            df['month']=range(maxidx, len(ddt))
            T=np.array(df.month)
            Q=np.array(df.Oil)
            return df, Q, T
        
        if Gas==True:
            ddt=pd.read_csv(data_path)
            ddt=ddt[np.isfinite(ddt.Gas)]
            maxidx=ddt[ddt.Oil==ddt.Oil.max()].index.values[0]
            df=ddt.iloc[maxidx:]
            df['month']=range(maxidx, len(ddt))
            T=np.array(df.month)
            Q=np.array(df.Gas)
            return df, Q, T
        
    elif start_from_max==False:
        if Oil==True:
            df=pd.read_csv(data_path)
            df=df[np.isfinite(df.Oil)]
            df['month']=range(len(df))
            T=np.array(df.month)
            Q=np.array(df.Oil)
            return df, Q, T
        
        if Gas==True:
            df=pd.read_csv(data_path)
            df=df[np.isfinite(df.Gas)]
            df['month']=range(len(df))
            T=np.array(df.month)
            Q=np.array(df.Gas)
            return df, Q, T
            
"""
#####################################################################################################
SINGLE WELL DATA FORMAT EXAMPLE (SINGLE WELL)
#####################################################################################################
path='C:/Users/Owner/Downloads/monthly-production.csv'
df, Q, T=single_well_ARPS_data_format(path, Oil=True, Gas=False)
"""

"""
#####################################################################################################
EXPONENTIAL (SINGLE WELL)
#####################################################################################################
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
#####################################################################################################
HYPERBOLIC (SINGLE WELL)
#####################################################################################################
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
#####################################################################################################
HARMONIC (SINGLE WELL)
#####################################################################################################
"""
def har_decline(Q, T, T_end=None, q_f=None, q_i=None, plot=False):
    if T_end!=None:
        if q_f!=None:
            raise Exception("One or the other dumb fuck")
    if q_f!=None:
        if T_end!=None:
            raise Exception("One or the other dumb fuck")         
    q_i=Q[0]
    def harmonic_decline(q_i):
        def har_dec(T, a_i):
            return q_i/(1+a_i*T)
        return har_dec
    har_dec=harmonic_decline(q_i)
    a_har, a_har_cov=curve_fit(har_dec, T, Q, method="trf")
    pred_obs_har=har_dec(T, a_har[0])
    if T_end!=None:
        T_to_end=range(T_end)
        total_pred_har=har_dec(T_to_end, a_har[0])
        T_end_obs=max(T)
        T_future=range(T_end_obs, T_end)
        future_pred_har=har_dec(T_future, a_har[0])
        if plot==True:
            fig, ax = plt.subplots(1, figsize=(16, 16))
            ax.set_title("Harmonic Decline Curve Analysis", fontsize=30)
            label_size = 20
            yed = [tick.label.set_fontsize(label_size) for tick in ax.yaxis.get_major_ticks()]
            xed = [tick.label.set_fontsize(label_size) for tick in ax.xaxis.get_major_ticks()]
            ax.scatter(T, Q, color="black", marker="x", s=250, linewidth=3)
            ax.set_xlabel("Time (mos.)", fontsize=25)
            ax.set_ylabel("Oil Rate (1000 STB/d)", fontsize=25)
            ax.plot(T_to_end, total_pred_har, color="red", linewidth=5, alpha=0.5, label="Harmonic")
            ax.ticklabel_format(fontsize=25)
            ax.legend(fontsize=25)
            plt.show()
        return pred_obs_har, total_pred_har, future_pred_har, a_har, a_har_cov
    elif q_f!=None:
        T_end_obs=max(T)
        T_iter=max(T)
        q_pred=q_f+1
        while q_pred>q_f:
            T_iter_range=range(T_iter+1)
            T_future_range=range(T_end_obs, T_iter+1)
            total_pred_har=har_dec(T_iter_range, a_har[0])
            future_pred_har=har_dec(T_future_range, a_har[0])
            q_pred=total_pred_har[T_iter]
            T_iter+=1
            T_to_end=T_iter_range
        if plot==True:
            fig, ax = plt.subplots(1, figsize=(16, 16))
            ax.set_title("Harmonic Decline Curve Analysis", fontsize=30)
            label_size = 20
            yed = [tick.label.set_fontsize(label_size) for tick in ax.yaxis.get_major_ticks()]
            xed = [tick.label.set_fontsize(label_size) for tick in ax.xaxis.get_major_ticks()]
            ax.scatter(T, Q, color="black", marker="x", s=250, linewidth=3)
            ax.set_xlabel("Time (mos.)", fontsize=25)
            ax.set_ylabel("Oil Rate (1000 STB/d)", fontsize=25)
            ax.plot(T_to_end, total_pred_har, color="red", linewidth=5, alpha=0.5, label="Harmonic")
            ax.ticklabel_format(fontsize=25)
            ax.legend(fontsize=25)
            plt.show()
        return pred_obs_har, total_pred_har, future_pred_har, a_har, a_har_cov
    else:
        if plot==True:
                fig, ax = plt.subplots(1, figsize=(16, 16))
                ax.set_title("Harmonic Decline Curve Analysis", fontsize=30)
                label_size = 20
                yed = [tick.label.set_fontsize(label_size) for tick in ax.yaxis.get_major_ticks()]
                xed = [tick.label.set_fontsize(label_size) for tick in ax.xaxis.get_major_ticks()]
                ax.scatter(T, Q, color="black", marker="x", s=250, linewidth=3)
                ax.set_xlabel("Time (mos.)", fontsize=25)
                ax.set_ylabel("Oil Rate (1000 STB/d)", fontsize=25)
                ax.plot(T, pred_obs_har, color="red", linewidth=5, alpha=0.5, label="Harmonic")
                ax.ticklabel_format(fontsize=25)
                ax.legend(fontsize=25)
                plt.show()
        return pred_obs_har, a_har, a_har_cov

"""
#####################################################################################################
ARPS DECLINE EXAMPLES (SINGLE WELL)
#####################################################################################################

#Observed only decline curves
exp_prediction, a_exp, a_exp_covariance=exp_decline(Q, T)  #this is an example for exponential decline
hyp_prediction, a_hyp, a_hyp_covariance=hyp_decline(Q, T) #this is an example for hyperbolic decline
har_prediction, a_har, a_har_covariance=har_decline(Q, T) #this is an example of harmonic decline

#End time given decline curves
exp_prediction, total_pred_exp, future_pred_exp, a_exp, a_exp_covariance= exp_decline(Q, T, T_end=70)
hyp_prediction, total_pred_hyp, future_pred_hyp, a_hyp, a_hyp_covariance= hyp_decline(Q, T, T_end=70)
har_prediction, total_pred_har, future_pred_har, a_har, a_har_covariance= har_decline(Q, T, T_end=55)

#Economic flow limit given decline curves
exp_prediction, total_pred_exp, future_pred_exp, a_exp, a_exp_covariance= exp_decline(Q, T, q_f=0.5)
hyp_prediction, total_pred_hyp, future_pred_hyp, a_hyp, a_hyp_covariance= hyp_decline(Q, T, q_f=1)
har_prediction, total_pred_har, future_pred_har, a_har, a_har_covariance= har_decline(Q, T, q_f=15/1000)

#Failed due to T_end and q_f being defined
exp_prediction, a_exp, a_exp_covariance=exp_decline(Q, T, T_end=70, q_f=1) #this is an example of harmonic decline
hyp_prediction, a_hyp, a_hyp_covariance=hyp_decline(Q, T, T_end=70, q_f=1) #this is an example of harmonic decline
har_prediction, a_har, a_har_covariance=har_decline(Q, T, T_end=70, q_f=1) #this is an example of harmonic decline

#Exp with plot
exp_prediction, a_exp, a_exp_covariance=exp_decline(Q, T, plot=True)  #this is an example for exponential decline
a,b,c,d,e=exp_decline(Q, T, T_end=70, plot=True)  #this is an example for exponential decline
a,b,c,d,e=exp_decline(Q, T, q_f=0.6, plot=True)

#Hyp with plot
a,b,c=hyp_decline(Q, T, plot=True)  #this is an example for exponential decline
a,b,c,d,e=hyp_decline(Q, T, T_end=70, plot=True)  #this is an example for exponential decline
a,b,c,d,e=hyp_decline(Q, T, q_f=0.6, plot=True)

#Har with plot
a,b,c=har_decline(Q, T, plot=True)  #this is an example for exponential decline
a,b,c,d,e=har_decline(Q, T, T_end=70, plot=True)  #this is an example for exponential decline
a,b,c,d,e=har_decline(Q, T, q_f=0.6, plot=True)
"""

"""
#####################################################################################################
MULTI WELL DATA FORMATTING FUNCTION FOR ARPS
*note: to do any multi_well DCA's, user must run:
    from pyDCA import har_decline, exp_decline, hyp_decline
in addition to importing the multi well functions however you want
#####################################################################################################
"""
def multi_well_ARPS_data_format(data_path, Oil=True, Gas=False, start_from_max=False):
    if start_from_max==True:
        data=pd.read_csv(data_path)
        if Oil==True:
            df=data.drop(['Entity ID', 'API/UWI List', 'Gas (mcf)', 'Water (bbl)', 'Well Count', 'Days'], axis=1)
            df=df.rename(index=str, columns={'API/UWI':'API', 'Production Date':'ds', 'Liquid (bbl)':'Oil'})
            df=df[np.isfinite(df.Oil)]
    
            loopdf=[]
        
            for Oil, group in df[['Oil', 'ds', 'API']].groupby('API'):
                ddt=pd.DataFrame({'Oil':group.Oil, 'ds':group.ds, 'API':group.API})
                maxidx=int(ddt[ddt.Oil==ddt.Oil.max()].index.values[0])
                dt=ddt.iloc[maxidx:]
                dt['month']=range(maxidx, len(ddt))
                loopdf.append(dt)
            
            T=[]
            Q=[]
            
            for i in range(len(loopdf)):
                t=np.array(loopdf[i].month)
                q=np.array(loopdf[i].Oil)
                T.append(t)
                Q.append(q)
                
            return loopdf, Q, T
        
        if Gas==True:
            df=ddt.drop(['Entity ID', 'API/UWI List', 'Liquid (bbl)', 'Water (bbl)', 'Well Count', 'Days'], axis=1)
            df=df.rename(index=str, columns={'API/UWI':'API', 'Production Date':'ds', 'Gas (mcf)':'Gas'})
            df=df[np.isfinite(df.Gas)]
    
            loopdf=[]
        
            for Gas, group in df[['Gas', 'ds', 'API']].groupby('API'):
                ddt=pd.DataFrame({'Gas':group.Gas, 'ds':group.ds, 'API':group.API})
                maxidx=int(ddt[ddt.Gas==ddt.Gas.max()].index.values[0])
                dt=ddt.iloc[maxidx:]
                dt['month']=range(maxidx, len(ddt))
                loopdf.append(dt)
            
            T=[]
            Q=[]
            
            for i in range(len(loopdf)):
                t=np.array(loopdf[i].month)
                q=np.array(loopdf[i].Gas)
                T.append(t)
                Q.append(q)
                
            return loopdf, Q, T
        
    elif start_from_max==False:
        ddt=pd.read_csv(data_path)
        if Oil==True:
            df=ddt.drop(['Entity ID', 'API/UWI List', 'Gas (mcf)', 'Water (bbl)', 'Well Count', 'Days'], axis=1)
            df=df.rename(index=str, columns={'API/UWI':'API', 'Production Date':'ds', 'Liquid (bbl)':'Oil'})
            df=df[np.isfinite(df.Oil)]
    
            loopdf=[]
        
            for Oil, group in df[['Oil', 'ds', 'API']].groupby('API'):
                dt=pd.DataFrame({'Oil':group.Oil, 'ds':group.ds, 'API':group.API})
                loopdf.append(dt)
            
            T=[]
            Q=[]
            
            for i in range(len(loopdf)):
                loopdf[i]['month']=range(len(loopdf[i]))
                t=np.array(loopdf[i].month)
                q=np.array(loopdf[i].Oil)
                T.append(t)
                Q.append(q)
                
            return loopdf, Q, T
        
        if Gas==True:
            df=ddt.drop(['Entity ID', 'API/UWI List', 'Liquid (bbl)', 'Water (bbl)', 'Well Count', 'Days'], axis=1)
            df=df.rename(index=str, columns={'API/UWI':'API', 'Production Date':'ds', 'Gas (mcf)':'Gas'})
            df=df[np.isfinite(df.Gas)]
    
            loopdf=[]
        
            for Gas, group in df[['Gas', 'ds', 'API']].groupby('API'):
                dt=pd.DataFrame({'Gas':group.Gas, 'ds':group.ds, 'API':group.API})
                loopdf.append(dt)
            
            T=[]
            Q=[]
            
            for i in range(len(loopdf)):
                loopdf[i]['month']=range(len(loopdf[i]))
                t=np.array(loopdf[i].month)
                q=np.array(loopdf[i].Gas)
                T.append(t)
                Q.append(q)
                
            return loopdf, Q, T
            
    
"""
#####################################################################################################
MULTI WELL DATA FORMAT EXAMPLE
#####################################################################################################

path='C:/Users/Owner/Downloads/2wellsGrady Production Time Series.csv'
loopdf, Q, T=multi_well_ARPS_data_format(path, Oil=True, Gas=False)
"""

"""
#####################################################################################################
MULTI-WELL EXPONENTIAL
#####################################################################################################
"""
def multi_exp_decline(Q, T, T_end=None, q_f=None, q_i=None, plot=False):
    if T_end!=None:
        if q_f!=None:
            raise Exception("One or the other dumb fuck")
    if q_f!=None:
        if T_end!=None:
            raise Exception("One or the other dumb fuck") 
            
    loop_pred_obs_exp=[]
    loop_a_exp=[]
    loop_a_exp_cov=[]
    loop_total_pred_exp=[]
    loop_future_pred_exp=[]
    
    if T_end!=None:
        for i in range(len(Q)):
            exp_prediction, total_pred_exp, future_pred_exp, a_exp, a_exp_cov= exp_decline(Q[i], T[i], T_end=T_end, plot=plot)
            loop_a_exp.append(a_exp)
            loop_a_exp_cov.append(a_exp_cov)
            loop_pred_obs_exp.append(exp_prediction)
            loop_total_pred_exp.append(total_pred_exp)
            loop_future_pred_exp.append(future_pred_exp)
        return loop_pred_obs_exp, loop_total_pred_exp, loop_future_pred_exp, loop_a_exp, loop_a_exp_cov
    
    elif q_f!=None:
        for i in range(len(Q)):
            exp_prediction, total_pred_exp, future_pred_exp, a_exp, a_exp_cov= exp_decline(Q[i], T[i], q_f=q_f, plot=plot)
            loop_a_exp.append(a_exp)
            loop_a_exp_cov.append(a_exp_cov)
            loop_pred_obs_exp.append(exp_prediction)
            loop_total_pred_exp.append(total_pred_exp)
            loop_future_pred_exp.append(future_pred_exp)
        return loop_pred_obs_exp, loop_total_pred_exp, loop_future_pred_exp, loop_a_exp, loop_a_exp_cov
    else:
        for i in range(len(Q)):
            exp_prediction, a_exp, a_exp_cov=exp_decline(Q[i], T[i], plot=plot)  #this is an example for exponential decline
            loop_a_exp.append(a_exp)
            loop_a_exp_cov.append(a_exp_cov)
            loop_pred_obs_exp.append(exp_prediction)
    return loop_pred_obs_exp, loop_a_exp,  loop_a_exp_cov

"""
#####################################################################################################
MULTI-WELL HYPERBOLIC
#####################################################################################################
"""
def multi_hyp_decline(Q, T, T_end=None, q_f=None, q_i=None, plot=False):
    if T_end!=None:
        if q_f!=None:
            raise Exception("One or the other dumb fuck")
    if q_f!=None:
        if T_end!=None:
            raise Exception("One or the other dumb fuck") 
            
    loop_pred_obs_hyp=[]
    loop_a_hyp=[]
    loop_a_hyp_cov=[]
    loop_total_pred_hyp=[]
    loop_future_pred_hyp=[]
    
    if T_end!=None:
        for i in range(len(Q)):
            hyp_prediction, total_pred_hyp, future_pred_hyp, a_hyp, a_hyp_cov= hyp_decline(Q[i], T[i], T_end=T_end, plot=plot)
            loop_a_hyp.append(a_hyp)
            loop_a_hyp_cov.append(a_hyp_cov)
            loop_pred_obs_hyp.append(hyp_prediction)
            loop_total_pred_hyp.append(total_pred_hyp)
            loop_future_pred_hyp.append(future_pred_hyp)
        return loop_pred_obs_hyp, loop_total_pred_hyp, loop_future_pred_hyp, loop_a_hyp, loop_a_hyp_cov
    
    elif q_f!=None:
        for i in range(len(Q)):
            hyp_prediction, total_pred_hyp, future_pred_hyp, a_hyp, a_hyp_cov= hyp_decline(Q[i], T[i], q_f=q_f, plot=plot)
            loop_a_hyp.append(a_hyp)
            loop_a_hyp_cov.append(a_hyp_cov)
            loop_pred_obs_hyp.append(hyp_prediction)
            loop_total_pred_hyp.append(total_pred_hyp)
            loop_future_pred_hyp.append(future_pred_hyp)
        return loop_pred_obs_hyp, loop_total_pred_hyp, loop_future_pred_hyp, loop_a_hyp, loop_a_hyp_cov
    else:
        for i in range(len(Q)):
            hyp_prediction, a_hyp, a_hyp_cov=hyp_decline(Q[i], T[i], plot=plot)  #this is an example for hyponential decline
            loop_a_hyp.append(a_hyp)
            loop_a_hyp_cov.append(a_hyp_cov)
            loop_pred_obs_hyp.append(hyp_prediction)
    return loop_pred_obs_hyp, loop_a_hyp,  loop_a_hyp_cov

"""
#####################################################################################################
MULTI-WELL HARMONIC
#####################################################################################################
"""
def multi_har_decline(Q, T, T_end=None, q_f=None, q_i=None, plot=False):
    if T_end!=None:
        if q_f!=None:
            raise Exception("One or the other dumb fuck")
    if q_f!=None:
        if T_end!=None:
            raise Exception("One or the other dumb fuck") 
            
    loop_pred_obs_har=[]
    loop_a_har=[]
    loop_a_har_cov=[]
    loop_total_pred_har=[]
    loop_future_pred_har=[]
    
    if T_end!=None:
        for i in range(len(Q)):
            har_prediction, total_pred_har, future_pred_har, a_har, a_har_cov= har_decline(Q[i], T[i], T_end=T_end, plot=plot)
            loop_a_har.append(a_har)
            loop_a_har_cov.append(a_har_cov)
            loop_pred_obs_har.append(har_prediction)
            loop_total_pred_har.append(total_pred_har)
            loop_future_pred_har.append(future_pred_har)
        return loop_pred_obs_har, loop_total_pred_har, loop_future_pred_har, loop_a_har, loop_a_har_cov
    
    elif q_f!=None:
        for i in range(len(Q)):
            har_prediction, total_pred_har, future_pred_har, a_har, a_har_cov= har_decline(Q[i], T[i], q_f=q_f, plot=plot)
            loop_a_har.append(a_har)
            loop_a_har_cov.append(a_har_cov)
            loop_pred_obs_har.append(har_prediction)
            loop_total_pred_har.append(total_pred_har)
            loop_future_pred_har.append(future_pred_har)
        return loop_pred_obs_har, loop_total_pred_har, loop_future_pred_har, loop_a_har, loop_a_har_cov
    else:
        for i in range(len(Q)):
            har_prediction, a_har, a_har_cov=har_decline(Q[i], T[i], plot=plot)  #this is an example for haronential decline
            loop_a_har.append(a_har)
            loop_a_har_cov.append(a_har_cov)
            loop_pred_obs_har.append(har_prediction)
    return loop_pred_obs_har, loop_a_har,  loop_a_har_cov

"""
#####################################################################################################
ARPS DECLINE EXAMPLES (MULTI-WELL)
#####################################################################################################
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

"""
#####################################################################################################
DATA FORMAT FOR FBPROPHET TIME SERIES FORECASTING DECLINE CURVE
#####################################################################################################
"""
def fbprophet_DCA_data_format(data_path, Oil=True, Gas=False, multi_wells=True):
    if multi_wells==True:
        if Oil==True:
            ddt=pd.read_csv(data_path)
            df=ddt.drop(['Entity ID', 'API/UWI List', 'Gas (mcf)', 'Water (bbl)', 'Well Count', 'Days'], axis=1)
            df=df.rename(index=str, columns={'API/UWI':'API', 'Production Date':'ds', 'Liquid (bbl)':'y'})
        elif Gas==True:
            ddt=pd.read_csv(data_path)
            df=ddt.drop(['Entity ID', 'API/UWI List', 'Liquid (bbl)', 'Water (bbl)', 'Well Count', 'Days'], axis=1)
            df=df.rename(index=str, columns={'API/UWI':'API', 'Production Date':'ds', 'Gas (mcf)':'y'})
        return df
    if multi_wells==False:
        if Oil==True:
            ddt=pd.read_csv(data_path)
            df=ddt.drop(['Gas', 'Water'], axis=1)
            df=df.rename(index=str, columns={'DateTime':'ds', 'Oil':'y'})
        elif Gas==True:
            ddt=pd.read_csv(data_path)
            df=ddt.drop(['Oil', 'Water'], axis=1)
            df=df.rename(index=str, columns={'DateTime':'ds', 'Gas':'y'})
        return df
"""
#####################################################################################################
EXAMPLE OF DATA FORMAT FOR FBPROPHET TIME SERIES FORECASTING DECLINE CURVE
#####################################################################################################
#MULTIPLE WELLS (multi_well=True automatically)
path='C:/Users/Owner/Downloads/2wellsGrady Production Time Series.csv'
df=fbprophet_DCA_data_format(path)

#SINGLE WELL (must set multi_well=False)
singlepath='C:/Users/Owner/Downloads/monthly-production.csv'
df=fbprophet_DCA_data_format(singlepath, multi_wells=False)
"""
"""
#####################################################################################################
FBPROPHET TIME SERIES FORECASTING DECLINE CURVE
#####################################################################################################
"""
def fbprophet_DCA(df, multi_wells=True, q_f=15, max_months=400, min_observed_prod_to_remove=30, show_plots=True, save_plots=False):
    if multi_wells==False:
        df['API']=1
#Your API number will simply return as 1 if data for a single well is inputted due to the way 
#the data must be formatted since downloading prod data for one well on DrillingInfo doesnt include API (at least the way Im downloading it)
    loopdf=[]
    APIdf=[]
    
    for y, group in df[['y', 'ds', 'API']].groupby('API'):
        dt=pd.DataFrame({'y':group.y, 'ds':group.ds})
        loopdf.append(dt)
        well=pd.DataFrame({'API':group.API})
        APIdf.append(well)
    
    
    fcast=[]
    prediction=[]
    maxidx=[]
    API_EUR=np.zeros((len(loopdf), 2))
    
    for i in range(len(loopdf)):
        loopdf[i]=loopdf[i].reset_index()
        midx=loopdf[i][loopdf[i].y==loopdf[i].y.max()].index.values[0]
        maxidx.append(midx)
        loopdf[i]=loopdf[i][maxidx[i]:]
        loopdf[i]=loopdf[i][loopdf[i].y>min_observed_prod_to_remove]
        loopdf[i]=loopdf[i][np.isfinite(loopdf[i].y)]
        loopdf[i]=loopdf[i].reset_index()
        loopdf[i]=loopdf[i].drop(['index', 'level_0'], axis=1)
       
        loopdf[i].y = np.log(loopdf[i].y)
    #df.y=pd.rolling_mean(df.y, window=10)
        m=Prophet()
        m.fit(loopdf[i])
        future = m.make_future_dataframe(periods=max_months, freq='M')
        fcst = m.predict(future)
        #m.plot(fcst);
        
        new=fcst[['ds', 'yhat', 'yhat_upper', 'yhat_lower']].copy()
        new['y']=loopdf[i].y
        new.y=np.exp(new.y)
        new.yhat=np.exp(new.yhat)
        new=new.set_index('ds')
        new=pd.DataFrame(new)
        new=new[new.yhat>q_f]
        fcast.append(new)
        
    #FINDING CUMULATIVE
        Ldf=len(loopdf[i])
        pred=fcast[i][0:Ldf]
        pred=pred[['y']].copy()
        yhat=fcast[i].yhat[Ldf:] 
        pred=pred.y.append(yhat)
        pred=pd.DataFrame(pred)
        pred['cum']=pred.cumsum()
        prediction.append(pred)
        prediction[i]=prediction[i].rename(columns={0:'Oil'})
        ult=max(prediction[i].cum)
        prediction[i]['EUR']=ult
        prediction[i]['API']=APIdf[i].API[0]
    #plotting
        if show_plots==True:
            plt.ion()
        elif show_plots==False:
            plt.ioff()
        fig, ax1 = plt.subplots()
        plot1,=ax1.plot(fcast[i].y[0:Ldf], label='actual')
        plot2,=ax1.plot(fcast[i].yhat, color='black', linestyle=':', label='forecasted')
        ax1.set_title(str(prediction[i].API[0])+' fbprophet Decline')
        ax1.set_ylabel('Oil Production (bbl)')
        ax1.set_xlabel('Date')
        ax1.legend(handles=[plot1, plot2])     
    #saving plot
        if save_plots==True:
            plt.savefig('fbprophet DCA'+str(APIdf[i].API[0])+'.png')    
       
        #Make something that ony contains API and EUR to put in production headers
        API_EUR[i, 0]=int(APIdf[i].API[0])
        API_EUR[i,1]=prediction[i].EUR[0]
        
    return API_EUR, fcast, prediction

"""
#####################################################################################################
EXAMPLES OF FBPROPHET TIME SERIES FORECASTING DECLINE CURVE
#####################################################################################################
#SINGLE WELL (must specify multi_wells=False) *API number will just be set to 1 due to the way single well
#production data is downloaded from Drilling Info
eur, forecast_outputs, total_prediction=fbprophet_DCA(df, multi_wells=False)

#MULTIPLE WELLS
eur, forecast_outputs, total_prediction=fbprophet_DCA(df)

"""

