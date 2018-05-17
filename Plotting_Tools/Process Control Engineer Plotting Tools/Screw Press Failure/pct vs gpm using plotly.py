# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:34:35 2018

@author: Owner
"""
import pandas as pd
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
plotly.tools.set_credentials_file(username='gagerussell', api_key='UKvdW4bzPZl6UFLUcFb4')

dateparse = lambda x: pd.datetime.strptime(x, '%d-%m-%Y %H:%M:%S')
df=pd.read_csv('C:/Users/Owner/Documents/International Paper/Machine_Failure/newfixed_compiled_sludge_data.csv',parse_dates=['Timestamp'], date_parser=dateparse)#, nrows=5376, skiprows=6)#, dtype={'E_SCREW_PR_MTR_LOAD(pct)':np.float, 'W_SCREW_PR_MTR_LOAD_x':np.float })
df=df.drop(columns=['Waste_Clarifier_Pit_Level(in)'])
df.Timestamp[df.Timestamp<'2017-12-21 00:00:00'] = df.Timestamp[df.Timestamp<'2017-12-21 00:00:00'].apply(lambda x: x + pd.DateOffset(years=1))
df=df.reset_index(drop=True)
df['n1pct']=df['n1_PRES_SLUDGE_FLO(pct)']*6
df['n2pct']=df['n2_PRES_SLUDGE_FLO(pct)']*6

df1=df[:100000]
df2=df[100000:200000]
df3=df[200000:300000]
df4=df[300000:400000]
df5=df[400000:500000]
df6=df[500000:600000]
df7=df[600000:]
Df=[df1, df2, df3, df4, df5, df6, df7]

for i in range(len(Df)):
    print('starting loop')
    n1gpm = go.Scattergl(
        x=Df[i].Timestamp,
        y=Df[i]['n1_PRES_SLUDGE_FLO(gpm)'],
        name = "n1 FLOW (gpm)",
        line = dict(color = '#17BECF'),
        opacity = 0.8)
    
    n1pct = go.Scattergl(
        x=Df[i].Timestamp,
        y=Df[i]['n1pct'],
        name = "n1 FLOW (pct)",
        line = dict(color = '#7F7F7F'),
        opacity = 0.8)
    
    data = [n1gpm,n1pct]

    layout = dict(
        title=str(i*100) + 'to' + str(100*(i+1)) + 'Sludge Press Time Series With Range-Slider',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=15,
                         label='15 min',
                         step='minute',
                         stepmode='backward'),
                    dict(count=60,
                         label='60 min',
                         step='minute',
                         stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(),
            type='date'
        )
    )

    print('plotting plot number: ' + str(i))
    fig = dict(data=data, layout=layout)
    py.iplot(fig, filename = str(i*100) + 'to' + str(100*(i+1)) + 'Sludge Press Time Series With Range-Slider')
    print('finished plotting plot number: ' + str(i))
