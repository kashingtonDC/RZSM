# SAR2RZSM

### Objective: 
Examine the relationship between radar backscatter ($\sigma$) observations, root zone soil moisture, and leaf water potential. 

<img src="sigma_v_SM.png" width="400">

### Theoretical Basis and Approach
Many Studies (e.g. Konings, 2017; Momen, 2017) show that $\Psi$ can be retrieved via passive microwave observations due to the sensitivity of microwaves to the dielectric constant of water. So, active radar (e.g. Sentinel SAR, which is also microwave radiation) should also be able to be used for soil moisture estimation.

However, there are many complicating factors that muddle this retrieval. These include (1) vegetation (2) landcover (3) precipitation events (4) soil types, and more. 

Solution: Use 

### Notes: 
only ascending orbits are used

### Data
1. SCAN sitelist: https://wcc.sc.egov.usda.gov/nwcc/yearcount?network=scan&counttype=statelist&state=
2. SCAN Site Map: https://www.wcc.nrcs.usda.gov/webmap/
2. Sentinel-1 C band SAR: 
        Data availability: 2014-10-03 – Present
        Repeat frequency: ~20 days (only ascending orbits)
        Cross polarization vs VV, HH
3. Soils data: SURGO, World Harmonized Soils Database? 


##### to get the SCAN data
1. Go to https://wcc.sc.egov.usda.gov/reportGenerator/ and build a query with desired columns. 

Once you set the params and hit enter, it's easy to 
auto generate a table of data for a given `site_id` using the following URL and substituting `site_id` where `2218` is:

https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultiTimeSeriesGroupByStationReport,metric/hourly/start_of_period/2218:CA:SCAN%7Cid=%22%22%7Cname/-35315,-11/SMS:-2:value,SMS:-4:value,SMS:-8:value,SMS:-20:value,SMS:-40:value

### To get the SSURGO data: 

1. Supply lat/lons in this relevant query:https://casoilresource.lawr.ucdavis.edu/soil_web/reflector_api/soils.php?what=mapunit&lon=-120.72&lat=40.62
2. Scrape the "ogc_fid" from the resulting html
3. submit that id into this url string: https://casoilresource.lawr.ucdavis.edu/soil_web/ssurgo.php?action=explain_mapunit&query_scale=500000&mukey=487690&ogc_fid=1533434

and voila


### Questions and TODO:

1. Sites of interest?
2. Time periods of interest? 
3. Sampling frequency? 
    daily? hourly? 
4. Polarizations? 

