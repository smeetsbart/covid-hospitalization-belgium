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

settings = \
   { "IFR"           : 0.0057#Infection to fatality ratio. 0.4% is number from WHO
   , 'R0'            : 5.2#Baseline reproduction number. 6.2 is value from original paper Maier & Brockmann
   , "t_infectious"  : 8#Amount of days that one is infectious. 8 is value from original paper Maier & Brockmann
   , "p_immune"      : 0.5#If you were already infected in previous wave, what is probability of being immune now.
   , 'p_ICU'         : 0.15#Given Hospitalization, chance of ending up in ICU
   , 'time_ICU'      : 14#Average time a patient spends in ICU before release/death/regular hospi
   , "delay_ICU"     : 5#Average time a patients spends in regular hospital before transfer to ICU
   , 'delay_death'   : 14#Average time delay in days between hospitalization and death.
   , "pop"           : 11606426#Total population (Belgium) as of 2020-10-31
   , "start_date"    : datetime.date( 2020, 9, 10)#Start date for wave / epidemy
   , "proj_date"     : datetime.date( 2020, 12, 25 )#Date to show projection of numbers
   }

def load_data( settings ):
   download_data.download_data(settings['start_date'])
   dbase = np.recfromcsv('tally.csv', encoding='UTF-8')
   return dbase

def fit_model( dbase, settings, do_plot=False, verbose=False, print_table=False ):

   sirm = SIRX.SIRXConfirmedModel()

   d_all = dbase['d_all']
   h_all = dbase['h_all']
   dstart = d_all[0]
   hstart = h_all[0]
   istart = dstart / settings['IFR']#Number of infecteds at start of wave

   hospitalization_fraction = hstart / istart#Fraction of infecteds at start of wave which was severe
   if hospitalization_fraction >= 1.0:
      raise ValueError("\nHospitalization fraction may not be larger than one! Your estimated IFR might be way too high?")
   immune_start = istart/settings['pop']*settings['p_immune']#Fraction of population immune at start of wave
   S_start = 1. - immune_start#Fraction of population susceptible at start of wave
   htotal = dbase['htot']
   hcurr = dbase['h']
   release = dbase['r']
   icu = dbase['icu']
   death = dbase['d']
   days = np.array([int(el) for el in dbase['dag']])
   #dates = [date_start+date_time.timedelta( days=int(i)-shift) for i in range(days)]
   day0 = days[0]
   shift=1
   days -= day0 - shift#Start at day 1, for log fit

   date_start = settings['start_date']
   date_end   = date_start + datetime.timedelta( days=int(days[-1])-shift )

   d2 = htotal[-1] - 2*htotal[-2] + htotal[-3]
   htotal_ext = htotal[-1] + (htotal[-1]-htotal[-2]) + d2/2.

   ifr = settings['IFR']
   scaling_factor = 1./hospitalization_fraction
   population=settings['pop']#Belgian population

   params_all = []
   day_c = np.linspace( days[0], (settings['proj_date']-settings['start_date']).days, 10000)

   Ndshift = 0
   dshifts = list(range(-Ndshift,1))
   diffs = []
   icudiffs = []
   ddiffs = []
   dshift = settings['delay_death']
   icushift = settings['delay_ICU']
   dscale = settings['IFR']  * scaling_factor

   #Loop normally only runs once. For loop only for back-fitting with bootstrapped data:
   for dmin in dshifts:
      xdata = days[:len(days)+dmin]
      ydata = (htotal*scaling_factor)[:len(days)+dmin]

      params = sirm.fit(xdata,ydata,max_nfev=20000
                       , N=population
                       , R0=settings['R0']
                       , S0 = S_start
                       , rho=1/settings['t_infectious']).params
      N = params['N']
      params_all.append(params)
      t = days
      tswitch = t[-1]
      tt = np.logspace(np.log(t[0]), np.log(day_c[-1]), 5000,base=np.exp(1))
      tt1 = tt[tt<=tswitch]
      tt2 = tt[tt>tswitch]
      result = sirm.SIRX(tt, scaling_factor*htotal[0],
                         params['eta'],
                         params['rho'],
                         params['kappa'],
                         params['kappa0'],
                         N,
                         params['I0_factor'],
                         S_start
                         )
      X = result[2,:]*N/scaling_factor
      I = result[1,:]*N/scaling_factor
      S = result[0,:]*N/scaling_factor
      Z = result[3,:]*N/scaling_factor

      Xshift = np.interp( tt-settings['time_ICU'], tt, X, left=X[0] )
      Hcurr = (X - Xshift)*settings['p_ICU'] + icu[0]

      ymod = np.interp( days, tt, X)
      icumod = np.interp( days, tt, Hcurr)
      dmod = np.interp( days, tt+dshift, X*dscale )

      diff = ( ymod[-1] - htotal[-1] ) / htotal[-1] * 100
      icudiffs.append( (icumod[-1] - icu[-1] )/icu[-1] * 100 )
      ddiffs.append( (dmod[-1] - death[-1]) / death[-1] * 100 )

      diffs.append(diff)

      residuals = htotal - ymod
      ss_res = np.sum( residuals**2)
      ss_tot = np.sum( (htotal - np.mean(htotal))**2)
      r_sq = 1 - (ss_res/ss_tot)

   peak_icu_tt = np.argmax(Hcurr)
   tt_peak = (tt+icushift)[peak_icu_tt]
   peak_date = date_start + datetime.timedelta(days=int( tt_peak ))

   if verbose:
      print(f" - Fit SIRX: r^2 = {r_sq:1.5f}")
      print(f" - Start date: {date_start}")
      print(f" - End date  : {date_end}")
      print(f" - 'Severe' infecteds: {hospitalization_fraction*100:1.2f}%")
      print(f" - Deaths at start of wave: {dstart:,}")
      print(f" - Susceptible population: {int(S_start*population):,}")
      print(f" - Peak infected number this wave: {int( I.max()*scaling_factor ):,}")
      print(f" - Total infecteds this wave {int(X.max()*scaling_factor):,} ")
      print(f" - Total infecteds: {int(X.max()*scaling_factor+dstart/ifr):,}")
      print(f" - Projected total wave deaths: {int( np.max(X*dscale) ):,}")
      print(f" - Projected total deaths: {dstart + int( np.max(X*dscale)):,}")
      print(f" - Latest ICU occupancy: {icu[-1]}")
      print(f" - Projected peak ICU: {int(max(Hcurr))} patients")
      print(f" - Projected peak Date: {peak_date}")
      print(f" - Estimated immune at {settings['start_date']} : {immune_start*100:1.2f}%")
      print(f" - Projected immune at {settings['proj_date']} : {(X.max()*scaling_factor+dstart/ifr)*settings['p_immune']/settings['pop']*100:1.2f}%")

   alpha = ufloat(params['eta'].value, params['eta'].stderr if params['eta'].stderr else 0.)
   beta = ufloat(params['rho'].value, params['rho'].stderr if params['rho'].stderr else 0.)
   k0 = ufloat(params['kappa0'].value,params['kappa0'].stderr if params['kappa0'].stderr else 0.)
   k = ufloat(params['kappa'].value, params['kappa'].stderr if params['kappa'].stderr else 0.)

   Puf = k0 / (  k+k0)
   R0_eff = alpha/( beta + k + k0 )
   Q = (k + k0) / (beta + k + k0)

   if print_table:
      print("\n")
      print("Parameter             & Estimate & Std. Error\\\\")
      print("\\hline")
      print(f"\\kappa_0              & {k0.n:1.3f} & {k0.s:1.3f} \\\\")
      print(f"\\kappa                & {k.n:1.3f} & {k.s:1.3f} \\\\")
      print(f"$P$                   & {Puf.n:1.3f} & {Puf.s:1.3f} \\\\")
      print(f"$Q$                   & {Q.n:1.3f} & {Q.s:1.3f} \\\\")
      print(f"R_{{0,\\mathrm{{eff}}}}    & {R0_eff.n:1.3f} & {R0_eff.s:1.3f} \\\\")
      print("\\hline")
      print("\n")

   results = {}
   results["icu_peak"] = int(max(Hcurr))
   results['Q'] = Q.n
   results['R0_eff'] = R0_eff.n
   results['P'] = Puf.n
   results['death_wave'] = int(np.max(X*dscale))
   results['death_total'] = dstart + results['death_wave']
   results['immune_before'] = immune_start
   results['immune_after'] = (X.max()*scaling_factor+dstart/ifr)*settings['p_immune']/settings['pop']
   results['rsq'] = r_sq
   results['d_peak'] =  tt_peak - days[-1]

   if do_plot:
      fig = plt.figure( figsize=(8,6))

      ax0,ax1 = fig.subplots(2, 1, gridspec_kw={'height_ratios': [1,1]})

      black = (0.2,0.2,0.2)
      ax0.yaxis.tick_right()

      first_saturday = [i for i in days if (date_start+datetime.timedelta(days = int(i-shift))).weekday()==5][0]

      #One tick on monday, first day of the week
      dayticks = range( first_saturday+2-7, int(day_c[-1])+7,7)
      dates_fmt = [date_start+datetime.timedelta(days = i-shift)  for i in dayticks]
      daylabels = [f"{el.day}/{el.month}" for el in dates_fmt]

      weekend = [(first_saturday+7*i,first_saturday+2+7*i) for i in range(20)]
      ymax = 1.1*X[-1]
      xlim = [ day_c[0]-1, day_c[-1] ]


      if xscale=='linear':
         for weekendi in weekend:
            ax0.fill_betweenx( [0,ymax]
                             , [weekendi[0], weekendi[0]]
                             , [weekendi[1], weekendi[1]]
                             , color=(0.2,0.2,0.5), alpha=0.05 )
         for day_i in range(dayticks[0],dayticks[-1]):
            ax0.plot([day_i,day_i],[0,ymax],'-k', alpha=0.04,lw=0.5)

      ax0.plot( days, hcurr, marker='o', mfc='None', mec='C0', label='$H$',ls='None', alpha=0.5)
      ax0.plot( tt+dshift, X*dscale, '-.', color=black,lw=2
              , label=f'$X_D$ ($\Delta=${dshift} days, IFR={settings["IFR"]*100:1.2f}%)')
      ax0.plot( days, htotal, 'o', label='$H_a$', ms=8, color='C0',mec='None')
      ax0.plot( tt, X, '-', color=black, label="$X$", lw=2)
      ax0.plot( days, release, '+',mew=3, color='C2', label='$R$',ms=9)
      ax0.plot( days, death, 'x', label='$D$', color='C3',mew=2)
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
      ymax = 1.2*max(Hcurr)
      if xscale=='linear':
         for weekendi in weekend:
            ax1.fill_betweenx( [0,ymax]
                             , [weekendi[0], weekendi[0]]
                             , [weekendi[1], weekendi[1]]
                             , color=(0.2,0.2,0.5), alpha=0.05 )
         for day_i in range(dayticks[0],dayticks[-1]):
            ax1.plot([day_i,day_i],[0,ymax],'-k', alpha=0.04,lw=0.5)

      ax1.plot( tt+icushift, Hcurr, ':', lw=3, color=black
              , label=f'$X_{{ICU}}$ ($\Delta$={settings["delay_ICU"]} days, t={settings["time_ICU"]} days, p={settings["p_ICU"]*100:1.0f}%)')
      ax1.plot( days, icu, '+', mew=3, color='C4', label='ICU', ms=9)

      ax1.set_xlabel('Date', fontsize=14)
      ax1.set_xscale(xscale)
      ax1.set_yscale(yscale)
      ax1.set_ylim( 0, ymax)
      ax1.set_ylabel("N", fontsize=14)
      ax1.legend(frameon=False, fontsize=12, loc=2, ncol=1)
      if xscale =='linear':
         ax1.set_xticks( [el for el in dayticks] )
         ax1.set_xticklabels(daylabels)
         ax1.set_xticks( range(dayticks[0],dayticks[-1]),minor=True )
      ax1.set_xlim( *xlim )

      plt.tight_layout()
      basename = f'fitted_models_{date_end}'
      pdfname = f'{basename}.pdf'
      pngname = f'{basename}.png'
      plt.savefig(pngname,dpi=300)
      plt.show()

   return results


if __name__ == "__main__":
   dbase = load_data( settings )
   results = fit_model( dbase, settings, do_plot=True,verbose=True,print_table=True)
   print(results)

