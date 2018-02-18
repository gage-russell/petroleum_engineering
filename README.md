# petroleum_engineering

<h2>Folder: Decline Curve Analysis</h2>
<p><h3><b>fbprophet_DCA2.0.py</b></h3></p>
<body>
  <p><b>Summary:</b><br>
  Does decline curve analysis for any number of wells downloaded as production time series csv from drillinginfo (download the production   time series csv from drillinginfo for any number of wells and leave it exactly as is). Only thing to change is path to file. The most     important final output is stored in the 'prediction' variable, which will contain a dataframe for each well in the csv. One forecast       figure and one DCA figure will be saved for each well.
   </P>
   <ul>
    <li>Libraries needed:</li> 
      <ul>
        <li>pandas</li> <li>matplotlib</li> <li>numpy</li> <li>fbprophet</li>
      </ul>
    <li>Data Format:</li> 
      <ul>
        <li>Production Time Series csv from DrillingInfo AS-IS (any number of wells)</li>
      </ul>
    <li>Uncomment:</li>
      <ul>
        <li>m.plot(fcst) to plot each forecast (not a good idea for lots of wells)</li>
      </ul>
    <li>Figures:</li>
      <ul>
        <li>DCA [API#].png are the final DCA (Actual+Forecast) figures created and saved by the program (example used two wells)</li>
        <li>fcst [API#].png are the forecast plots generated with fbprophet (example used two wells)</li>
      </ul>
    </ul>
</body>

<h2>Folder: Reservoir Fluid Characterization</h2>
<p><h3><b>Res_Fluid_Clustering.py</b></h3></p>
<body>
  <p><b>Summary:</b><br>
  Uses KMeans clustering to characterize reservoir fluid types. This particular example creates three clusters and then further divides     the cluster with the lowest average initial GOR because it is dealing specifically with oil wells in Grady County producing from the       Woodford. Folium is used to map each well on an interactive map in the browser. 
  </p>
  <ul>
    <li>Libraries needed:</li>
      <ul>
        <li>pandas</li> <li>matplotlib</li> <li>numpy</li> <li>sklearn</li> <li>folium</li>
      </ul>
    <li>Data Format:</li> 
      <ul>
        <li>Production Headers csv from DrillingInfo AS-IS (any number of wells)</li>
      </ul>
  <li>Figures:</li>
    <ul>
      <li>res_fluid_clustermap.html is the output of the example ran using wells in Grady County clustered into four fluid types</li>
    </ul>
  </ul>
  </body>
  
