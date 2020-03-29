#!/usr/bin/env python3
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import numpy as np
import os
fit = optimize.curve_fit
xscale = 'log'
yscale = 'log'

#Models to fit
def logistic( x, L, k, x0 ):
   return L / ( 1 + np.exp( -k*(x-x0) ) )

def exp( x, k, x0, y0):
   return y0*np.exp( k*(x-x0) )

def quadratic( x, a0, a1, a2):
   return a0 + a1 * x + a2*x**2

def si(  x, N0, k, b ):
   return N0 * np.exp( k*x )*np.exp( -b * x**2 )

def linear( x, a, b):
   return a*x+b

def do_fit( f, x, y, **kwargs ):
   verbose=kwargs.pop('verbose', True)
   name = kwargs.pop('name', f.__name__)
   popt, pcov = fit( f, x, y, **kwargs)
   perr = np.sqrt( np.diag(pcov) )
   residuals = y - f( x, *popt )
   ss_res = np.sum( residuals**2)
   ss_tot = np.sum( (y - np.mean( y ))**2 )
   r_squared = 1 - (ss_res / ss_tot)
   if verbose:
      print(f" - Fit {name:20}| r^2 = {r_squared:1.5f}")
   return popt, perr, r_squared

dbase = np.recfromcsv('tally.csv', encoding='UTF-8')

htotal = dbase['htot']
hcurr = dbase['h']
release = dbase['r']
death = dbase['d']
days = dbase['dag']
day0 = days[0]
days -= day0 - 1#Start at day 1, for log fit

d2 = htotal[-1] - 2*htotal[-2] + htotal[-3]
htotal_ext = htotal[-1] + (htotal[-1]-htotal[-2]) + d2/2.

plog,elog,rsqlog = do_fit( logistic, days, htotal, p0 = ( 2*htotal[-1], 1.0, 25. ) )
pexp,eexp,rsqexp = do_fit( exp, days[:5], htotal[:5], p0 = ( 2.0, 10, 1. ))
pquad, equad, rsqquad = do_fit( quadratic, days, htotal, p0 = (0.,0.5,1.0))
psi, esi, rsqsi = do_fit( si, days, htotal, bounds = ([ 0., 0., 0. ],[ 1e3, 2.0, 1.0 ] ) )
pll, ell, rsqll = do_fit( linear, np.log(days[5:]), np.log( htotal[5:] ),p0=( 1.,2. ), name='loglog' )

if xscale == 'log':
   day_c = np.linspace( days[0]/2, days[-1]*2, 10000)
else:
   day_c = np.linspace( days[0]-4, days[-1]+8, 10000)

model_log = logistic( day_c, *plog)
model_exp = exp( day_c, *pexp)
model_quad = quadratic( day_c, *pquad)
model_si = si( day_c, *psi)
model_ll = np.exp( linear( np.log(day_c), *pll) )

tomorrow = days[-1] + 1

N_tomorrow = { 'logistic':logistic( tomorrow, *plog)
             , 'exp':exp( tomorrow, *pexp)
             , "quadratic":quadratic( tomorrow, *pquad)
             , "si":si(tomorrow, *psi)
             , 'loglog':np.exp( linear( np.log(tomorrow), *pll ))}

for modname in N_tomorrow:
   print(f"Delta N {modname:20} : {int(N_tomorrow[modname])-htotal[-1]}")

print(f"logistic cross over date: {int( plog[2] )+day0-1}")
print(f"logistic plateau: {int( plog[0] )}")
print(f'power law exponent: {pll[0]:1.3f}')

dayticks = range( 0, 28,4)
lday = [(lab-2+day0)%31+1 for lab in dayticks]
lmonth = [(lab-2+day0)//31+3 for lab in dayticks]
daylabels = [f"{day}/{month}" for day,month in zip(lday,lmonth)]

fig = plt.figure( figsize=(6,5))
ax = fig.add_subplot(111)

plt.plot( day_c, model_log, '--k',lw=2, label='Logistic')
plt.plot( day_c, model_exp,'-', color='gray', label='Exponential')
plt.plot( day_c, model_ll,'-', color='k', label='Power law')
plt.plot( day_c, model_si, '-.k', color='gray', label="SI-X", lw=2)
plt.plot( days, htotal, 'o', label='H+D+R', ms=8, color='C0')
plt.plot( days, death, 'x', label='D', color='C3',mew=2)
plt.plot( days, release, '+',mew=3, color='C2', label='R',ms=9)

plt.xlabel('Day', fontsize=14)
plt.xscale(xscale)
plt.yscale(yscale)
plt.ylim( 0 if yscale=='linear' else 10, 1.2*model_log.max() )
plt.ylabel("N", fontsize=14)
plt.legend(frameon=False, fontsize=12, loc=0)
if xscale =='linear':
   ax.set_xticks( [el for el in dayticks],daylabels )
   ax.set_xticks( range(dayticks[0],dayticks[-1]),minor=True )
plt.xlim( day_c[0],day_c[-1] )
plt.tight_layout()
fname = f'fitted_models_{days[-1]+day0-1}.png'
plt.savefig(fname, dpi=300)
os.system(f'gwenview {fname}')







