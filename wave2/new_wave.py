import matplotlib.pyplot as plt
import SIRX
import numpy as np
import pickle, datetime
import fit_models

s = fit_models.settings
s["proj_date"] = datetime.date( 2021, 3, 1 )#Sufficently far in the future for any peak
dbase = fit_models.load_data( s )

mx=0
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,6.5))
for R0i,IFRi,ax in zip( [2.5,4.0],[1.25/100,0.7/100],[ax1,ax2]):

   s['R0'] = R0i
   s['IFR'] = IFRi
   data = fit_models.fit_model( dbase, s, do_plot=False)
   Q = data['Q']

   sirm = SIRX.SIRXConfirmedModel()


   days = np.arange(1,100)
   tt = np.logspace(np.log(days[0]), np.log(days[-1]), 1000,base=np.exp(1))

   eta = data['eta']
   rho = data['rho']
   kappa = data['kappa']
   kappa0 = data['kappa0']
   N = data['N']
   I0_factor = data['I0_factor']
   S0 = data['Sf']#Immune from current wave
   #S0 = 1.

   waves = [ 1., data['S0'], data['Sf'] ]
   results = []
   N0 = data['X0'] * N
   print(f"N0={N0}")

   for i,wave_i in enumerate(waves):
      result = sirm.SIRX( tt , N0, eta, rho, kappa, kappa0, N, I0_factor, wave_i)
      X = result[2,:]*N
      I = result[1,:]*N
      S = result[0,:]*N
      Z = result[3,:]*N
      results.append(result)

      ax.plot( tt, X/data['scaling_factor'], label=f"$S_0$={wave_i:1.2f} (wave {i+1})",lw=2 )
      mx = max( mx, np.max( X/data['scaling_factor']) )
   ax.set_title(f"$R_0$={R0i:1.2f}, $IFR$={IFRi*100:1.2f}%, $Q$={Q:1.2f}, $X_{{a,0}}$={int(N0):,}",fontsize=15)
   ax.set_xlabel("Day", fontsize=16)
   ax.set_ylabel('$H_a$', fontsize=16)
   #ax.set_yscale('log')
   ax.legend(frameon=False, fontsize=13,loc=2)

ax1.set_ylim( 0, 1.2*mx)
ax2.set_ylim( 0, 1.2*mx)

plt.tight_layout()
plt.savefig("simulated_third_wave_comparison.png",dpi=300)


