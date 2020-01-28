# SAR2RZSM

### Objective: 
Examine relationships between radar backscatter, soil moisture, leaf water potential, and other hydrobiophysical parameters

<img src="sigma_v_SM.png" width="400">

## Introduction
This study seeks to examine empirical relations between remote sensing data and biophysical paramters using in Situ SCAN sites as ground truth.

## Methods and Preprocessing

### Pedotransfer Functions 
	Are used to transform measured soil moisture to leaf water potential
    Unified North American Soil Map
    https://daac.ornl.gov/NACP/guides/NACP_MsTMIP_Unified_NA_Soil_Map.html
    

### Filtering (S1, Precipitation, SM )
	Only Ascending S1 orbits with VV polarization are used
    Eliminated sites in Alaska, Puerto Rico, Hawaii from SCAN db
    Filter out soil moisture values with preceding precipitation determined by PRISM 

## Data

### Soil Moisture (SCAN):
	Soil Climate Analysis Network data, root zone, surface soil moisture calculated by averaging over depth intervals. 
	List of SCAN sites: https://wcc.sc.egov.usda.gov/nwcc/yearcount?network=scan&counttype=statelist&state=
	Query builder: https://wcc.sc.egov.usda.gov/reportGenerator/ and build a query with desired columns. 

### Sentinel-1 C band SAR (EE): 
    Data availability: 2014-10-03 – Present
    Ascending Orbits Only
    Polarizations: VV, HV
    Naming convention: (https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/naming-conventions)

### PRISM precipitation data (EE):
	2-day sums are calculated at each SCAN site to filter out SM observations during saturated conditions

### Landsat B1 - B7
    

### MODIS / Landsat / Proba NDVI  


### Soil Texture:
    Unified North American Soil Map
    https://daac.ornl.gov/NACP/guides/NACP_MsTMIP_Unified_NA_Soil_Map.html
    Not using: Harmonized World Soil Database
    http://www.fao.org/soils-portal/soil-survey/soil-maps-and-databases/harmonized-world-soil-database-v12/en/
    