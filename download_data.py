import json
import numpy as np
import requests,datetime

dates = []

d0 = 15
def date_to_index( datestr, start_date = datetime.date( 2020,3,d0 ) ):
   date = [int(el) for el in datestr.split('-')]
   date = datetime.date( *date )
   dates.append(date)
   index = (date - start_date).days
   return index

def fname( url ):
   return url.split('/')[-1]

def load_data( url ):
   dr = requests.get(url, allow_redirects=True)
   open( fname( url ), 'wb').write(dr.content)
   f = open(fname(url),'r', encoding='latin-1')
   return json.load(f)

def parse_data( data, entryname):
   entries = {}
   for entry in data:
      entry['t'] = date_to_index(entry['DATE'])
      entries[entry['t']] = entries.get(entry['t'],0) + entry[entryname]
   return list(entries.keys()), list( entries[i] for i in list(entries.keys()) )


#URLs to pull data in from:
d_url = "https://epistat.sciensano.be/Data/COVID19BE_MORT.json"
h_url = "https://epistat.sciensano.be/Data/COVID19BE_HOSP.json"

entries = {}

ddata = load_data(d_url)
hdata = load_data(h_url)

days_d, deaths = parse_data( load_data(d_url), 'DEATHS')
d = np.cumsum(deaths)
d = [di for i,di, in zip(days_d,d) if i >= 0]

hdata = load_data(h_url)

print(f" - Downloaded json data between {min(dates)} and {max(dates)}")

days, h = parse_data( hdata, "TOTAL_IN")

days, r = parse_data( hdata, "NEW_OUT")

days, icu = parse_data( hdata, "TOTAL_IN_ICU")
r = np.cumsum(r)

htot = [ hi+ri+di for hi,ri,di in zip( h,r,d ) ]

data = np.c_[ np.array(days)+d0, h, r, d, htot, icu  ]
data = np.array(data, dtype=int)
header = "Dag,H,R,D,Htot,ICU"
np.savetxt( 'tally.csv', data, delimiter=',', header=header,fmt='%i')
