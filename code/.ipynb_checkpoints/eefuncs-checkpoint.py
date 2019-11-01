
'''
This module contains functions to handle conversion between ee.ImageCollection class and numpy.ndarray class

Processing flow is:
ee.ImageCollection --> pandas.DataFrame --> numpy.ndarray

'''

# Libs

import numpy as np
import pandas as pd

# Functions

def array_from_col(col,band,res,bounds,year,month,day):
    
    '''
    Transform an ee.ImageCollection class to a numpy array
    '''
    
    # get the lat lon and add the band and scale by the appropriate factor (0.0001 for landsat)
    band_name = col.select(band).median()
    latlon = ee.Image.pixelLonLat().addBands(band_name).multiply(0.0001)

    # apply reducer to list
    latlon = latlon.reduceRegion(
      reducer=ee.Reducer.toList(),
      geometry=bounds,
      maxPixels=1e13,
      scale=res)
    
    data = np.array((ee.Array(latlon.get(band)).getInfo()))
    lats = np.array((ee.Array(latlon.get("latitude")).getInfo()))
    lons = np.array((ee.Array(latlon.get("longitude")).getInfo()))
    
    arr = array_from_coords(data,lats,lons)
    
    return (arr)

def array_from_coords(data,lats,lons):
    
    '''
    Return a numpy array (ie cartesian product) from lats, lons, and data values
    '''
    
    # get the unique coordinates
    uniqueLats = np.unique(lats)
    uniqueLons = np.unique(lons)

    # get number of columns and rows from coordinates
    ncols = len(uniqueLons)    
    nrows = len(uniqueLats)

    # determine pixelsizes
    ys = uniqueLats[1] - uniqueLats[0] 
    xs = uniqueLons[1] - uniqueLons[0]

    # create an array with dimensions of image
    arr = np.zeros([nrows, ncols], np.float32) #-9999

    # fill the array with values
    counter =0
    for y in range(0,len(arr),1):
        for x in range(0,len(arr[0]),1):
            if lats[counter] == uniqueLats[y] and lons[counter] == uniqueLons[x] and counter < len(lats)-1:
                counter+=1
                arr[len(uniqueLats)-1-y,x] = data[counter] 
                
    return arr