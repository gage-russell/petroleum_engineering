# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 20:55:55 2018

@author: Owner
"""

import pandas as pd
import folium as folium
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Uses Production Headers File from drillinginfo for any number of wells
all_wells=pd.read_csv("C:/Users/Owner/Documents/Python Scripts/data/WellsTableGrady.csv")
ddt=all_wells.loc[:,["Surface Hole Latitude (WGS84)", "Surface Hole Longitude (WGS84)", "Cum Oil", "API14", "First Test Oil Gravity", "First Test GOR"]] 
ddt=ddt.rename(index=str, columns={"Surface Hole Latitude (WGS84)":"lat", "Surface Hole Longitude (WGS84)":"lon", "Cum Oil":"cum_oil", "First Test Oil Gravity":"gravity", "First Test GOR":"GOR"})
ddt['idx']=range(0, len(ddt))
ddt.dropna(axis=0, how='any', inplace=True)
ddt=ddt[ddt.GOR!=0]

#3 initial clusters
X=ddt.as_matrix(columns=ddt.columns[4:6])
y_pred=KMeans(n_clusters=3).fit_predict(X)
plt.scatter(X[:,0], X[:,1], c=y_pred)
df=ddt.copy()
df['cluster']=y_pred
df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

#THIS IS TO break one of the clusters into 2 clusters if you are aiming for a particular fluid classifications
#SPECIFICALLY FOR DF1 being lowest GOR AND I AIM TO CLOSELY RESEMBLE McCain's fluid classification
X2=df1.as_matrix(columns=df1.columns[4:6])
y2_pred=KMeans(n_clusters=2).fit_predict(X2)
plt.scatter(X2[:,0], X2[:,1], c=y2_pred)
df1['cluster']=None
df1['cluster']=y2_pred
df1_1=df1[df1.cluster==0]
df1_2=df1[df1.cluster==1]
#df1_1.cluster.replace(to_replace=0, value=2)      #THIS LINE TO CHANGE CLUSTER NUMBER OF FIRST SUBcluster if needed
df1_2.cluster.replace(to_replace=1, value=3)
df1=df1_1.copy()
df4=df1_2.copy()

#look at std devs for GOR: (optional)
GOR_sd=[df1.GOR.std(), df2.GOR.std(), df3.GOR.std(), df4.GOR.std()]
print(GOR_sd)
#look at std devs for gravity:  (optional)
gravity_sd=[df1.gravity.std(), df2.gravity.std(), df3.gravity.std(), df4.gravity.std()]
print(gravity_sd)

#Create interactive maps of the wells classified by fluid type that include the details of each well
res_fluid_clustermap= folium.Map(location=[35.03340, -97.90032],              #The wells in this example are in Grady County, OK
                        zoom_start=10,
                        tiles="CartoDB dark_matter")

for row in df1.itertuples():
        lat=str(row.lat)
        lon=str(row.lon)
        grav=str(row.gravity)
        GOR=str(row.GOR)
        idx=str(row.idx)
        popup_text = "<br>latitude: "+lat+"<br> longitude: "+lon+"<br> API gravity: "+grav+"<br> initial GOR "+GOR+"<br> index: "+idx
        folium.CircleMarker(location=(row.lat,row.lon), radius=2, fill=True, fill_color='blue', color='blue', popup=popup_text).add_to(res_fluid_clustermap)

for row in df2.itertuples():
        lat=str(row.lat)
        lon=str(row.lon)
        grav=str(row.gravity)
        GOR=str(row.GOR)
        idx=str(row.idx)
        popup_text = "<br>latitude: "+lat+"<br> longitude: "+lon+"<br> API gravity: "+grav+"<br> initial GOR "+GOR+"<br> index: "+idx
        folium.CircleMarker(location=(row.lat,row.lon), radius=2, fill=True, fill_color='red', color='red', popup=popup_text).add_to(res_fluid_clustermap)

for row in df3.itertuples():
        lat=str(row.lat)
        lon=str(row.lon)
        grav=str(row.gravity)
        GOR=str(row.GOR)
        idx=str(row.idx)
        popup_text = "<br>latitude: "+lat+"<br> longitude: "+lon+"<br> API gravity: "+grav+"<br> initial GOR "+GOR+"<br> index: "+idx
        folium.CircleMarker(location=(row.lat,row.lon), radius=2, fill=True, fill_color='green', color='green', popup=popup_text).add_to(res_fluid_clustermap)

for row in df4.itertuples():
        lat=str(row.lat)
        lon=str(row.lon)
        grav=str(row.gravity)
        GOR=str(row.GOR)
        idx=str(row.idx)
        popup_text = "<br>latitude: "+lat+"<br> longitude: "+lon+"<br> API gravity: "+grav+"<br> initial GOR "+GOR+"<br> index: "+idx
        folium.CircleMarker(location=(row.lat,row.lon), radius=2, fill=True, fill_color='white', color='white', popup=popup_text).add_to(res_fluid_clustermap)


res_fluid_clustermap.save("res_fluid_clustermap.html")