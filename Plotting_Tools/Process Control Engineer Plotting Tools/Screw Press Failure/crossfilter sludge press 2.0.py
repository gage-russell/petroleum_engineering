# -*- coding: utf-8 -*-
"""
Created on Wed May 16 22:05:51 2018

@author: Owner
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go


dateparse = lambda x: pd.datetime.strptime(x, '%d-%m-%Y %H:%M:%S')
dff=pd.read_csv('C:/Users/Owner/Documents/International Paper/Machine_Failure/newfixed_compiled_sludge_data.csv',parse_dates=['Timestamp'], date_parser=dateparse)#, nrows=5376, skiprows=6)#, dtype={'E_SCREW_PR_MTR_LOAD(pct)':np.float, 'W_SCREW_PR_MTR_LOAD_x':np.float })
dff=dff.drop(columns=['Waste_Clarifier_Pit_Level(in)'])
dff.Timestamp[dff.Timestamp<'2017-12-21 00:00:00'] = dff.Timestamp[dff.Timestamp<'2017-12-21 00:00:00'].apply(lambda x: x + pd.DateOffset(years=1))
dff=dff.reset_index(drop=True)
dff['n1pct']=dff['n1_PRES_SLUDGE_FLO(pct)']*6
dff['n2pct']=dff['n2_PRES_SLUDGE_FLO(pct)']*6


df=dff[:200000]

app = dash.Dash()

available_indicators = df.columns

app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='n1_PRES_SLUDGE_FLO(gpm)'
            ),
            dcc.RadioItems(
                id='crossfilter-xaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='crossfilter-yaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='n1pct'
            ),
            dcc.RadioItems(
                id='crossfilter-yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),

    html.Div([
        dcc.Graph(id='crossfilter-indicator-scatter')
    ], style={'width': '95%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        #dcc.Graph(id='crossfilter-indicator-scatter'),
        dcc.Graph(id='x-time-series'),
        dcc.Graph(id='y-time-series'),
    ], style={'display': 'inline-block', 'width': '95%', 'margin-top':5, 'padding': '0 20'})
])




def create_time_series(column_name, axis_type, title):
    return {
        'data': [go.Scattergl(
            x=df['Timestamp'],
            y=df[column_name],
            mode='line'
        )],
        'layout': {
            'height': 300,
            'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
            'annotations': [{
                'x': 0, 'y': 5, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                'text': title
            }],
            'yaxis': {'type': 'linear' if axis_type == 'Linear' else 'log', 'gridwidth':2, 'zerolinecolor':'#969696', 'zerolinewidth':4},
            'xaxis' : { 'rangeselector' : 
                {'buttons':[{'count':15, 'label':'15 min', 'step':'minute', 'stepmode':'backward'},
                            {'count':60, 'label':'60 min', 'step':'minute', 'stepmode':'backward'},
                            {'step':'all'}]}, 'rangeslider':{}, 'type':'date', 'mirror':True, 'gridcolor':'#bdbdbd',
                                'gridwidth':2,
                                'zerolinecolor':'#969696',
                                'zerolinewidth':4}
        }
    }
            

def create_time_multiseries(column_names, axis_type, title):
    return {
        'data': [go.Scattergl(
            x=df['Timestamp'],
            y=df['n1_PRES_SLUDGE_FLO(gpm)'],
            mode='lines+markers'
        ), go.Scattergl(
            x=df['Timestamp'],
            y=df['n1pct'],
            mode='line'
        )],
        'layout': { 'showlegend':False,
            'height': 400,
            'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
            'annotations': [{
                'x': 0, 'y': 5, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                'text': title
            }],
            'yaxis': {'type': 'linear' if axis_type == 'Linear' else 'log'},
            'xaxis' : { 'rangeselector' : 
                {'buttons':[{'count':15, 'label':'15 min', 'step':'minute', 'stepmode':'backward'},
                            {'count':60, 'label':'60 min', 'step':'minute', 'stepmode':'backward'},
                            {'step':'all'}]}, 'rangeslider':{}, 'type':'date', 'hovermode':'closest',
                                'gridcolor':'#bdbdbd',
                                'gridwidth':2,
                                'zerolinecolor':'#969696',
                                'zerolinewidth':4}
        }
    }

@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
    [dash.dependencies.Input('crossfilter-yaxis-column', 'value')])
def update_main(yaxis_column_name):
    return create_time_multiseries(['n1_PRES_SLUDGE_FLO(gpm)', yaxis_column_name], 'Linear', 'main')

@app.callback(
    dash.dependencies.Output('x-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-type', 'value')])
def update_y_timeseries(hoverData, yaxis_column_name, axis_type):
    title = yaxis_column_name
    return create_time_series(yaxis_column_name, axis_type, title)


@app.callback(
    dash.dependencies.Output('y-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value')])
def update_x_timeseries(xaxis_column_name, axis_type):
    title=xaxis_column_name
    return create_time_series(xaxis_column_name, axis_type, title)

if __name__ == '__main__':
    app.run_server()