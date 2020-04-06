#!/usr/bin/env python3
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from uncertainties import ufloat
import numpy as np
import datetime
import os
fit = optimize.curve_fit
import SIRX
import download_data
xscale = 'linear'
yscale = 'linear'

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

sirm = SIRX.SIRXConfirmedModel()

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
icu = dbase['icu']
death = dbase['d']
days = np.array([int(el) for el in dbase['dag']])
day0 = days[0]
shift = 5
days -= day0 - shift#Start at day 1, for log fit
print(f" - Starting day: {day0-shift} March")
print(f' - Infection start around {day0-shift-7} March')

date_start = datetime.date( 2020,3,day0-shift-7)

d2 = htotal[-1] - 2*htotal[-2] + htotal[-3]
htotal_ext = htotal[-1] + (htotal[-1]-htotal[-2]) + d2/2.

plog,elog,rsqlog = do_fit( logistic, days, htotal, p0 = ( 2*htotal[-1], 1.0, 25. ) )
pexp,eexp,rsqexp = do_fit( exp, days[:3], htotal[:3], p0 = ( 2.0, 10, 1. ))
pquad, equad, rsqquad = do_fit( quadratic, days, htotal, p0 = (0.,0.5,1.0))
psi, esi, rsqsi = do_fit( si, days, htotal, bounds = ([ 0., 0., 0. ],[ 1e3, 2.0, 1.0 ] ) )
pll, ell, rsqll = do_fit( linear, np.log(days[5:-5]), np.log( htotal[5:-5] ),p0=( 1.,2. ), name='power law' )
pd, ed, rsqd = do_fit( linear, np.log(days[5:-5]), np.log( death[5:-5]),p0=( 1.,2. ), name='power law death' )

tshift = -2*pquad[0]/pquad[1]
print(f" - tshift = {tshift:1.3f} days")

scaling_factor =20#10pct is hospitalized
population=11576427#Belgian population
params_all = []
day_c = np.linspace( days[0], days[-1]+14, 10000)

Ndshift = 10
dshifts = list(range(-Ndshift,1))
diffs = []
icudiffs = []

for dmin in dshifts:
   xdata = days[:len(days)+dmin]
   ydata = (htotal*scaling_factor)[:len(days)+dmin]
   params = sirm.fit(xdata,ydata,maxfev=10000,N=population).params
   N = params['N']
   params_all.append(params)
   t = days
   tswitch = t[-1]
   tt = np.logspace(np.log(t[0]), np.log(day_c[-1]), 1000,base=np.exp(1))
   tt1 = tt[tt<=tswitch]
   tt2 = tt[tt>tswitch]
   result = sirm.SIRX(tt, scaling_factor*htotal[0],
                      params['eta'],
                      params['rho'],
                      params['kappa'],
                      params['kappa0'],
                      N,
                      params['I0_factor'],
                      )
   X = result[2,:]*N/scaling_factor
   I = result[1,:]*N/scaling_factor
   S = result[0,:]*N/scaling_factor
   Z = result[3,:]*N/scaling_factor


   Xshift = np.interp( tt-12, tt, X, left=X[0] )
   Hcurr = (X - Xshift)*0.15

   ymod = np.interp( days, tt, X)
   icumod = np.interp( days, tt, Hcurr)
   diff = ( ymod[-1] - htotal[-1] ) / htotal[-1] * 100
   icudiffs.append( (icumod[-1] - icu[-1] )/icu[-1] * 100 )
   diffs.append(diff)

   residuals = htotal - ymod
   ss_res = np.sum( residuals**2)
   ss_tot = np.sum( (htotal - np.mean(htotal))**2)
   r_sq = 1 - (ss_res/ss_tot)
   print(f" - Shift = {-dmin} days, SIRX   | r^2 = {r_sq:1.5f}")



tomorrow = days[-1]+1

N_tomorrow = { 'logistic':logistic( tomorrow, *plog)
             , 'exp':exp( tomorrow, *pexp)
             , "quadratic":quadratic( tomorrow, *pquad)
             , "si":si(tomorrow, *psi)
             , "SIR-X": np.interp( [tomorrow], tt, X )[0]
             , 'power law':np.exp( linear( np.log(tomorrow), *pll )) }

for modname in N_tomorrow:
   print(f"Delta N {modname:20} : {int(N_tomorrow[modname])-htotal[-1]}")

alpha = ufloat(params['eta'].value, params['eta'].stderr)
beta = ufloat(params['rho'].value, params['rho'].stderr)
k0 = ufloat(params['kappa0'].value,params['kappa0'].stderr)
k = ufloat(params['kappa'].value, params['kappa'].stderr)

Puf = k0 / (  k+k0)

R0_eff = alpha/( beta + k + k0 )

Q = (k + k0) / (beta + k + k0)
print("\n")

print("Parameter             & Estimate & Std. Error\\\\")
print("\\hline")
print(f"\\mu                   & {pll[0]:1.3f} & {ell[0]:1.3f} \\\\")
print(f"\\mu_d                 & {pd[0]:1.3f} & {ed[0]:1.3f} \\\\")
print(f"\\kappa_0              & {k0.n:1.3f} & {k0.s:1.3f} \\\\")
print(f"\\kappa                & {k.n:1.3f} & {k.s:1.3f} \\\\")
print(f"$P$                   & {Puf.n:1.3f} & {Puf.s:1.3f} \\\\")
print(f"$Q$                   & {Q.n:1.3f} & {Q.s:1.3f} \\\\")
print(f"R_{{0,\\mathrm{{eff}}}}    & {R0_eff.n:1.3f} & {R0_eff.s:1.3f} \\\\")
print("\\hline")
print("\n")


model_log = logistic( day_c, *plog)
model_exp = exp( day_c, *pexp)
model_quad = quadratic( day_c, *pquad)
model_si = si( day_c, *psi)
model_ll = np.exp( linear( np.log(day_c), *pll) )
model_d = np.exp( linear( np.log(day_c),*pd))

dayticks = range( -1, int(day_c[-1])+10,7)
lday = [(lab-1+day0-shift)%31+1 for lab in dayticks]
lmonth = [(lab-1+day0-shift)//31+3 for lab in dayticks]
daylabels = [f"{day}/{month}" for day,month in zip(lday,lmonth)]

date_end   = datetime.date( 2020
                          , (days[-1]-1+day0-shift)//31+3
                          , (days[-1]-1+day0-shift)%31+1 )

fig = plt.figure( figsize=(5,8))

ax0,ax1,ax2 = fig.subplots(3, 1, gridspec_kw={'height_ratios': [2, 2, 1]})

black = (0.2,0.2,0.2)
ax0.yaxis.tick_right()

weekend = [(14+7*i,16+7*i) for i in range(10)]
ymax = 1.2*X[-1]
xlim = [ day_c[0], day_c[-1] ]

if xscale=='linear':
   for weekendi in weekend:
      ax0.fill_betweenx( [0,ymax]
                       , [weekendi[0]-day0+shift, weekendi[0]-day0+shift]
                       , [weekendi[1]-day0+shift, weekendi[1]-day0+shift]
                       , color=(0.2,0.2,0.5), alpha=0.05 )
   for day_i in range(dayticks[0],dayticks[-1]):
      ax0.plot([day_i,day_i],[0,ymax],'-k', alpha=0.04,lw=0.5)

ax0.plot( day_c, model_ll,'--', color=black, label=f'$\propto t^{{{pll[0]:1.2f}}}$')
ax0.plot( tt, X, '-', color=black, label="$X$", lw=2)
ax0.plot( days, htotal, 'o', label='$H_a$', ms=8, color='C0',mec='None')
ax0.plot( days, release, '+',mew=3, color='C2', label='$R$',ms=9)
ax0.annotate( f"$t_0=$ {date_start}", (0.65,0.05),xycoords = 'axes fraction' )
ax0.set_xscale(xscale)
ax0.set_yscale(yscale)
ax0.set_ylim( 0, ymax)
ax0.set_ylabel("N", fontsize=14)
ax0.legend(frameon=False, fontsize=12, loc=2, ncol=1)
if xscale =='linear':
   ax0.set_xticks( [el for el in dayticks] )
   ax0.set_xticklabels(daylabels)
   ax0.set_xticks( range(dayticks[0],dayticks[-1]),minor=True )
ax0.set_xlim( *xlim )

ax1.yaxis.tick_right()
ymax = 1.2*0.15*X[-1]
if xscale=='linear':
   for weekendi in weekend:
      ax1.fill_betweenx( [0,ymax]
                       , [weekendi[0]-day0+shift, weekendi[0]-day0+shift]
                       , [weekendi[1]-day0+shift, weekendi[1]-day0+shift]
                       , color=(0.2,0.2,0.5), alpha=0.05 )
   for day_i in range(dayticks[0],dayticks[-1]):
      ax1.plot([day_i,day_i],[0,ymax],'-k', alpha=0.04,lw=0.5)

ax1.plot( tt+2, X*0.15, '-', color=black,lw=2, label='$X_D$ (2 days, 15%)')
ax1.plot( tt, Hcurr, ':', lw=3, color=black, label='$X_{ICU}$ (12 Days, 15%)')
ax1.plot( days, death, 'x', label='$D$', color='C3',mew=2)
ax1.plot( days, icu, '+', mew=3, color='C4', label='ICU', ms=9)

ax1.set_xlabel('Date', fontsize=14)
ax1.set_xscale(xscale)
ax1.set_yscale(yscale)
ax1.set_ylim( 0, 1.2*0.15*X[-1])
ax1.set_ylabel("N", fontsize=14)
ax1.legend(frameon=False, fontsize=12, loc=2, ncol=1)
if xscale =='linear':
   ax1.set_xticks( [el for el in dayticks] )
   ax1.set_xticklabels(daylabels)
   ax1.set_xticks( range(dayticks[0],dayticks[-1]),minor=True )
ax1.set_xlim( *xlim )

xlim = (dshifts[0]-0.5,dshifts[-1]+0.5)
hlines = [10*i for i in range(9)]
for hline in hlines:
   ax2.plot( xlim,[hline,hline],'-k', alpha=0.1,lw=0.5)

ax2.plot( dshifts, diffs,'-',color='C0',alpha=0.5)
ax2.plot( dshifts, diffs,marker='o',color='C0',ls='None')
ax2.plot( dshifts, icudiffs,'-',color='C4',alpha=0.5)
ax2.plot( dshifts, icudiffs, marker='+', color='C4',ls='None',ms=9,mew=3)
ax2.set_ylabel("Error (%)", fontsize=14)
ax2.set_xlabel(f"Day (from {date_end})", fontsize=14)
ax2.set_yticks(hlines[::2])
ax2.set_xticks(dshifts)
ax2.set_xlim(xlim)

plt.tight_layout()
basename = f'fitted_models_{date_end}'
pdfname = f'{basename}.pdf'
pngname = f'{basename}.png'
plt.savefig(pngname,dpi=300)
plt.show()







