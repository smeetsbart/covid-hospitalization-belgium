import json
import numpy as np
import requests,datetime

dates = []

def date_to_index( datestr, start_date):
   date = [int(el) for el in datestr.split('-')]
   date = datetime.date( *date )
   dates.append(date)
   index = (date - start_date).days
   return index

def fname( url ):
   return url.split('/')[-1]


def fill_zeros( entries ):
   for i in range( min(entries.keys()),max(entries.keys()) ):
      if i not in entries:
         entries[i] = 0
   return entries

def load_data( url ):
   dr = requests.get(url, allow_redirects=True)
   open( fname( url ), 'wb').write(dr.content)
   f = open(fname(url),'r', encoding='latin-1')
   return json.load(f)

def parse_data( data, entryname, start_date, filter={}):
   entries = {}
   for entry in data:
      proceed = True
      for filterentry in filter:
         if filterentry not in entry:
            proceed = False
         else:
            proceed = proceed and (entry[filterentry] in filter[filterentry])
      entry['t'] = date_to_index(entry['DATE'], start_date)
      if proceed:
         entries[entry['t']] = entries.get(entry['t'],0) + entry[entryname]
      else:
         entries[entry['t']] = entries.get(entry['t'],0) + 0
   entries = fill_zeros( entries )
   return list(entries.keys()), list( entries[i] for i in list(entries.keys()) )

def download_data(start_date, fname = 'tally.csv'):

   #URLs to pull data in from:
   d_url = "https://epistat.sciensano.be/Data/COVID19BE_MORT.json"
   h_url = "https://epistat.sciensano.be/Data/COVID19BE_HOSP.json"

   entries = {}

   ddata = load_data(d_url)
   hdata = load_data(h_url)

   #ddict = {}
   #for agegroup in [[],['0-24','25-44'],['45-64'],['65-75'],['75-84','85+']]:
      #filter = {"AGEGROUP":agegroup} if len(agegroup) > 0 else {}
      #days_d, deaths = parse_data( load_data(url), "DEATHS", filter=filter)
      #d = [di for i,di in zip( days_d, np.cumsum(deaths)) if i >= 0]
      #ddict['']

   days_d, deaths = parse_data( load_data(d_url), 'DEATHS', start_date
                              , filter={})
   d_all = np.cumsum( deaths )
   d = [di for i,di, in zip(days_d,deaths) if i >= 0]
   d = np.cumsum(d)
   d_all = [di for i,di in zip(days_d, d_all) if i>=0]
   Nd = len(d)

   days_d, dy = parse_data( load_data(d_url), 'DEATHS', start_date
                              , filter={'AGEGROUP':['0-24','25-44']})
   dy = [di for i,di, in zip(days_d,dy) if i >= 0]
   dy = np.cumsum(dy)
   assert( len(dy) == Nd )

   days_d, dm = parse_data( load_data(d_url), 'DEATHS', start_date
                                  , filter={'AGEGROUP':['45-64']})
   dm = [di for i,di, in zip(days_d,dm) if i >= 0]
   dm = np.cumsum(dm)
   assert( len(dm) == Nd)

   days_r, dr = parse_data( load_data(d_url), 'DEATHS', start_date
                                  , filter={'AGEGROUP':['65-74']})
   dr = [di for i,di, in zip(days_d,dr) if i >= 0]
   dr = np.cumsum(dr)
   assert( len(dr) == Nd)

   days_d, do = parse_data( load_data(d_url), 'DEATHS', start_date
                                  , filter={'AGEGROUP':['75-84','85+']})
   do = [di for i,di, in zip(days_d,do) if i >= 0]
   do = np.cumsum(do)
   assert( len(do) == Nd)


   hdata = load_data(h_url)

   print(f" - Downloaded json data between {min(dates)} and {max(dates)}")

   days, h = parse_data( hdata, "TOTAL_IN", start_date)
   h = [hi for i,hi in zip(days,h) if i >= 0]

   assert( len(h)==Nd )

   days, r = parse_data( hdata, "NEW_OUT",start_date)
   r_all = np.cumsum(r)
   r_all = [ri for i,ri in zip(days, r_all) if i>=0]
   r = [ri for i,ri in zip(days,r) if i >= 0]

   days, icu = parse_data( hdata, "TOTAL_IN_ICU", start_date)
   icu = [icui for i,icui in zip(days,icu) if i >= 0]
   assert( len(icu) == Nd)
   r = np.cumsum(r)
   assert(len(r) == Nd)

   days = [i for i in days if i >= 0]
   assert(len(days) == Nd)

   htot = [ hi+ri+di for hi,ri,di in zip( h,r,d ) ]
   h_all = [ hi+ri+di for hi,ri,di in zip( h,r_all, d_all)]

   data = np.c_[ np.array(days)+start_date.day, h, r, d, htot, icu, dy, dm, dr, do, d_all,h_all  ]
   data = np.array(data, dtype=int)
   header = "Dag,H,R,D,Htot,ICU,Dy,Dm,Dr,Do,d_all,h_all"
   np.savetxt( fname, data, delimiter=',', header=header,fmt='%i')


if __name__ == "__main__":
   download_data( datetime.date( 2020,9,15 ) )

#days = list(entries.keys())
#deaths = np.cumsum(list( entries[i] for i in days ))

