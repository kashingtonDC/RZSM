import sys
import glob
import os
import pandas as pd
import ee
import xml.etree.ElementTree
import requests
import re

ee.Initialize()

def get_site_lat_lons(site_num, stations_csv):
	df = pd.read_csv(stations_csv)
	site = df[df['site_name'].str.contains(site_no)]
	
	lat, lon = site['lat'].values, site['lon'].values
	return lat[0], lon[0]

# grab the stations
csvs = glob.glob("*.csv")

# grab the site Ids
data_dir = os.path.join(os.getcwd(),"data")
txts = glob.glob(os.path.join(data_dir,"*.txt"))
stations_csv = csvs[0]

good_sites = 0
bad_sites = 0

for i in txts:
	try:
		site_no = ''.join(c for c in i if c.isdigit())

		# Using the lat / longs, create a small buffer and bbox, submit this to NRCS / USDA query string to get the "areasymbol"
		lat, lon = get_site_lat_lons(site_no,stations_csv)
		pt = ee.Geometry.Point([lon, lat])
		buffer_size = 10 # meters
		area = pt.buffer(buffer_size)
		bounds = area.bounds()

		# Extract upper left and lower right coordinates for the query 
		ul = bounds.getInfo()['coordinates'][0][0]
		lr = bounds.getInfo()['coordinates'][0][2]

		# query the webpage and get the map unit id
		query_string = "https://sdmdataaccess.nrcs.usda.gov/Spatial/SDMNAD83Geographic.wfs?Service=WFS&Version=1.0.0&Request=GetFeature&Typename=MapunitPoly&BBOX={},{},{},{}".format(ul[0], ul[1],lr[0], lr[1])

		req = requests.request('GET', query_string)
		xml = req.text
		soil_code = re.findall(r'areasymbol(.*?)areasymbol',xml)[0]
		mapunit = re.findall(r'>(.*?)<',soil_code)[0]

		print(mapunit)

		# Now parse the sand/silt/clay db to get the fractions 
		texture_file = [x for x in os.listdir(os.getcwd()) if x.endswith('ascii')][0]

		with open(texture_file) as f:
			for line in f:
				if mapunit in line:
					good_sites+= 1
					print(line)
				else:
					bad_sites+=1
	except:
		print(i + " FAILED") 
		bad_sites+=1

print(good_sites)
print(bad_sites)