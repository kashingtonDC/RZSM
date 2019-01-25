# SAR2RZSM

### Objective: 
Examine relationships between radar backscatter, soil moisture, leaf water potential, and other hydrobiophysical parameters

<img src="sigma_v_SM.png" width="400">

## Introduction
The determination of both Leaf water potential ($\Psi$) and root zone soil moisture from microwave satellite observations has been demonstrated by many studies. These parameters can be retrieved due to the sensitivity of the microwave spectrum to the dielectric constant of water. Studies (1,2) show that can be retrieved via passive microwave observations. While useful, the practical application ofpassive microwave retrievals are inhibited by coarse resolution (ref). Active radar (e.g. Sentinel SAR) offers the potential to retrieve measurements of RZSM at much higher resolution. However, a number of factors complicate this this process, including vegetation, double bounces, and more. 

This study seeks to examine empirical relations between backscatter and biophysical paramters using in Situ SCAN sites as ground truth.

## Methods

## Preprocessing

### Pedotransfer Functions 
	Are used to transform measured soil moisture to leaf water potential
	`mu_fract.ascii` is a lookup table relating each soil map unit (State_ID Site_ID format) to %clay, %silt, %sand
	see: http://www.soilinfo.psu.edu/index.cgi?soil_data&conus&data_cov&fract&methods
	TOOD: Add reference 

### Filtering (S1, Precipitation, SM )
	Only Ascending S1 orbits are used
	Currently, VV polarization and HV have been tested. Number of S1 Overpass differs
	Only soil moisture data without prior 3-day precipitation is used

## Data

### Soil Moisture (SCAN):
	Soil Climate Analysis Network data, root zone, surface soil moisture calculated by averaging over depth intervals. 
	List of SCAN sites: https://wcc.sc.egov.usda.gov/nwcc/yearcount?network=scan&counttype=statelist&state=
	Query builder: https://wcc.sc.egov.usda.gov/reportGenerator/ and build a query with desired columns. 

### Sentinel-1 C band SAR (EE): 
    Data availability: 2014-10-03 – Present
    Ascending Orbits Only
    Polarizations: VV, HV

### PRISM precipitation data (EE):
	Daily, ... 5-day sums are calculated at each SCAN site to filter out SM observations during saturated conditions

### Soil Texture (Soil units: USDA, Soil texture: Penn State):
	Workflow: Access soilDB R package and submit bbox query to get the soil "mapunit"

	Then use this lookup table which has sand, silt, clay fractions for each map unit: 
	http://www.soilinfo.psu.edu/index.cgi?soil_data&conus&data_cov&fract&datasets&lam
    
## Notes

### Questions and TODO:
1. Sites of interest?
2. Time periods of interest? 
3. Sampling frequency? 
    daily? hourly? 
4. Polarizations? - In general, HV polarization should be best. Why is this again?
