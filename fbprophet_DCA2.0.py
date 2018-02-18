# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 18:12:51 2018

@author: Owner
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 13:42:57 2018

@author: Owner
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fbprophet import Prophet

ddt1=pd.read_csv('C:/Users/Owner/Downloads/2wellsGrady Production Time Series.csv')

df=ddt1.copy()
df=df.drop(['Entity ID', 'API/UWI List', 'Gas (mcf)', 'Water (bbl)', 'Well Count', 'Days'], axis=1)
df=df.rename(index=str, columns={'API/UWI':'API', 'Production Date':'ds', 'Liquid (bbl)':'y'})

#grouped=df.groupby('API')

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
for i in range(len(loopdf)):
    loopdf[i]=loopdf[i].reset_index()
    midx=loopdf[i][loopdf[i].y==loopdf[i].y.max()].index.values[0]
    maxidx.append(midx)
    loopdf[i]=loopdf[i][maxidx[i]:]
    loopdf[i]=loopdf[i][loopdf[i].y>50]
    loopdf[i]=loopdf[i][np.isfinite(loopdf[i].y)]
    loopdf[i]=loopdf[i].reset_index()
    loopdf[i]=loopdf[i].drop(['index', 'level_0'], axis=1)
   
    loopdf[i].y = np.log(loopdf[i].y)
#df.y=pd.rolling_mean(df.y, window=10)
    m=Prophet()
    m.fit(loopdf[i])
    future = m.make_future_dataframe(periods=400, freq='M')
    fcst = m.predict(future)
    #m.plot(fcst);
    
    new=fcst[['ds', 'yhat', 'yhat_upper', 'yhat_lower']].copy()
    new['y']=loopdf[i].y
    new.y=np.exp(new.y)
    new.yhat=np.exp(new.yhat)
    new=new.set_index('ds')
    new=pd.DataFrame(new)
    new=new[new.yhat>15]
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
 