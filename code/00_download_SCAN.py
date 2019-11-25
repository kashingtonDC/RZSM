import sys
import glob
import os
import pandas as pd
import re

def read_file(filename):
    with open(filename) as f:
        contents = f.readlines()

    data = []
    for line in contents:
        if line[0:1] == "#":
            continue
        else:
            data.append(line)

    headers = [x.replace("Soil Moisture Percent","smp").replace(" ","_") for x in data[0].split(",")]
    cols = [x.strip("\n").split(",") for x in data[1:]]

    df = pd.DataFrame(cols, columns = headers)
    
    return df

def get_site_lat_lons(site_num, stations_csv):
    df = pd.read_csv(stations_csv)
    site = df[df['site_name'].str.contains(site_no)]
    
    lat, lon = site['lat'].values, site['lon'].values
    return lat[0], lon[0]


csvs = glob.glob("../*.csv")
stations_csv = csvs[0]
df = pd.read_csv(csvs[0])

names = list(df.site_name)
states = list(df.state)

site_ids = []
for i, x in enumerate(names):
    site_id = re.findall("\d+", names[i])[-1]
    site_ids.append(site_id)

# setup data dirs
cwd = os.getcwd()
data_dir = os.path.join(cwd, "../data")
if os.path.exists(data_dir):
	pass
else:
	os.mkdir(data_dir)

outfns = []
for i in site_ids:
	outfns.append(os.path.join(data_dir,i+".txt"))

for i, j in enumerate(site_ids):
    print(site_ids[i], states[i])
    query_string = '''https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultiTimeSeriesGroupByStationReport,metric/hourly/start_of_period/{}:{}:SCAN%7Cid=%22%22%7Cname/-35315,-11/SMS:-2:value,SMS:-4:value,SMS:-8:value,SMS:-20:value,SMS:-40:value '''.format(site_ids[i], states[i])
    
    command = '''curl {} --output {}'''.format(query_string, outfns[i])
    
    print(command)
    os.system(command)
