# SAR2RZSM

### Objective: 
Correlate radar backscatter ($\sigma$) observations to root zone soil moisture. 

### Theoretical Basis and Approach
Leaf water potential ($\Psi$) should correlate with RZSM. 
Studies (1,2) show that $\Psi$ can be retrieved via passive microwave observations.
This is due to the sensitivity of the microwave spectrum to the dielectric constant of water. 
So, active radar (e.g. Sentinel SAR) should also be able to retrieve measurements of 

### Notes: 
only ascending orbits are used

### Data
1. List of SCAN sites: https://wcc.sc.egov.usda.gov/nwcc/yearcount?network=scan&counttype=statelist&state=
2. Sentinel-1 C band SAR: 
        Data availability: 2014-10-03 – Present
        

#### to get the SCAN data
1. Go to https://wcc.sc.egov.usda.gov/reportGenerator/ and build a query with desired columns. 

Once you set the params and hit enter, it's easy to 
auto generate a table of data for a given `site_id` using the following URL and substituting `site_id` where `2218` is:

https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultiTimeSeriesGroupByStationReport,metric/hourly/start_of_period/2218:CA:SCAN%7Cid=%22%22%7Cname/-35315,-11/SMS:-2:value,SMS:-4:value,SMS:-8:value,SMS:-20:value,SMS:-40:value


        
### Questions and TODO:

1. Sites of interest?
2. Time periods of interest? 
3. Sampling frequency? 
    daily? hourly? 
4. Polarizations? 

