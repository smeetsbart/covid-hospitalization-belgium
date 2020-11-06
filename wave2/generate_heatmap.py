import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import fit_models
import copy
import multiprocessing
import datetime
import pickle

s0 = fit_models.settings

Nj = 20
Ni = 20

R0 = np.linspace(1.75,6.25,Ni )
IFR = np.linspace( 0.001,0.014,Nj )

PICU = np.zeros( ( len(R0),len(IFR) ) )
RSQ = np.zeros( np.shape(PICU) )
Q = np.zeros(np.shape(PICU))
DICU = np.zeros(np.shape(PICU))

dbase = fit_models.load_data( s0 )

I,J = np.meshgrid( range(Ni), range(Nj) )
N = I.size
print(f" - Performing model fits for {N} different conditions")

def process_ij( ij ):
   s = copy.deepcopy( s0 )
   i = ij[0]
   j = ij[1]
   s['IFR'] = IFR[j]
   s['R0'] = R0[i]
   s["proj_date"] = datetime.date( 2021, 3, 1 )#Sufficently far in the future for any peak
   return fit_models.fit_model( dbase, s, do_plot=False)

ijlist = [(i,j) for i,j in zip(I.flatten(),J.flatten())]
num_cores = multiprocessing.cpu_count()
rlist = Parallel(n_jobs=num_cores)(delayed(process_ij)(ij) for ij in ijlist)

#Single thread version if parallel would not work.
#rlist = []
#count = 0
#for i,j in zip(I.flatten(),J.flatten()):
   #s = copy.deepcopy( s0 )
   #s['IFR'] = IFR[j]
   #s['R0'] = R0[i]
   #s["proj_date"] = datetime.date( 2021, 3, 1 )#Sufficently far in the future for any peak
   #rlist.append( process_ij( (i,j) ) )
   #pct_done = count / N * 100
   #print(f" - Processed {count}/{N} parameter sets. {pct_done:1.2f}%")
   #count += 1

rdict = {}
for idx,rval in enumerate( rlist ):
   rdict[(I.flatten()[idx],J.flatten()[idx])] = rval

data_to_save = [ R0, IFR, rdict ]
with open('heatmap_results.pickle','wb') as fob:
   data = pickle.dump( data_to_save, fob )
