import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import copy
import datetime
import pickle
from pylab import cm

#Only used for specific contours. Generally we should not smooth anything
def smooth_image( image, sigma=0):
   if sigma > 0:
      from scipy import ndimage
      smoothed_image = ndimage.gaussian_filter(image, sigma=sigma)
   else:
      smoothed_image = image
   return smoothed_image

def T(val):
   return np.transpose(val)

th_rsq = 0.97

with open('heatmap_results.pickle','rb') as fob:
   data = pickle.load(fob)

#Unpack the data:
R0,IFR,rdict = data[0],data[1],data[2]

PICU = np.zeros( ( len(R0),len(IFR) ) )
RSQ = np.zeros( np.shape(PICU) )
RSQi = np.zeros( np.shape(PICU) )
Q = np.zeros(np.shape(PICU))
DICU = np.zeros(np.shape(PICU))
P = np.zeros(np.shape(PICU))
Reff = np.zeros(np.shape(PICU))
If = np.zeros(np.shape(PICU))

for i,R0i in enumerate(R0):
   for j,IFRi in enumerate(IFR):
      results = rdict[(i,j)]
      PICU[i,j] = results['icu_peak'] if results['rsq'] > th_rsq else np.nan
      RSQ[i,j] = results['rsq']
      RSQi[i,j] = results['rsqicu'] if results['rsq'] > th_rsq else np.nan
      Q[i,j] = results['Q'] if results['rsq'] > th_rsq else np.nan
      DICU[i,j] = results['d_peak'] if results['rsq'] > th_rsq else np.nan
      Reff[i,j] = results['R0_eff'] if results['rsq'] > th_rsq else np.nan
      If[i,j] = results['immune_after'] if results['rsq'] > th_rsq else np.nan
      P[i,j] = results['P'] if results['rsq'] > th_rsq else np.nan

PICU = np.ma.masked_where(np.isnan(PICU),PICU)
DICU = np.ma.masked_where(np.isnan(DICU),DICU)
Q = np.ma.masked_where(np.isnan(Q),Q)
P = np.ma.masked_where(np.isnan(P),P)
Reff = np.ma.masked_where(np.isnan(Reff),Reff)
RSQi = np.ma.masked_where(np.isnan(RSQi),RSQi)
If = np.ma.masked_where( np.isnan(If), If)

imax,jmax=np.unravel_index( np.argmax(RSQ), np.shape(RSQ) )
print(f"Peak value in r^2 was {np.max(RSQ)} for R0 = {R0[imax]} and IFR={IFR[jmax]}")
R0_set = R0[imax]
IFR_set = IFR[jmax]

#Figure of Peak ICU occupation
fig = plt.figure(figsize=(6,5))
ax=fig.add_subplot(111)
im = ax.pcolormesh(R0, IFR*100, T(PICU), cmap='RdYlGn_r',vmax=PICU.max())
CS = ax.contour(R0,IFR*100, T(PICU), levels=[1500,1550,1600,1650], colors='k')
ax.clabel(CS, inline=1, fontsize=10,fmt='%1.0f')
#CS2 = ax.contour( R0, IFR*100, T(smooth_image(RSQi,sigma=2)), levels = [0.97], colors='w')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("$N_{ICU}$",fontsize=16)
ax.set_xlabel( "$R_0$", fontsize=16 )
ax.set_ylabel( "IFR (%)", fontsize=16)
plt.tight_layout()
plt.savefig('heatmap_ICU_r0_ifr.png', dpi=300)


#Figure of days from today until peak ICU occupation
fig = plt.figure(figsize=(6,5))
ax=fig.add_subplot(111)
im = ax.pcolormesh(R0, IFR*100, T(DICU/7.), cmap='RdYlGn_r')
CS = ax.contour(R0,IFR*100, T(DICU/7.), levels=[0,1,2,3,4], colors='k')
ax.clabel(CS, inline=1, fontsize=10,fmt='%1.0f')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("$T_p$ (weeks)",fontsize=16)
ax.set_xlabel( "$R_0$", fontsize=16 )
ax.set_ylabel( "IFR (%)", fontsize=16)
plt.tight_layout()
plt.savefig('heatmap_Tp_r0_ifr.png', dpi=300)

#Figure of estimated quarantine probabilities
fig = plt.figure(figsize=(6,5))
ax=fig.add_subplot(111)
im = ax.pcolormesh(R0, IFR*100, T(Q), cmap='RdYlGn_r')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("$Q$",fontsize=16)
CS = ax.contour(R0,IFR*100, T(Q), levels=[0.3,0.4,0.5,0.6,0.7,0.8,0.9], colors='k')
ax.clabel(CS, inline=1, fontsize=10,fmt='%1.1f')
ax.set_xlabel( "$R_0$", fontsize=16 )
ax.set_ylabel( "IFR (%)", fontsize=16)
plt.tight_layout()
plt.savefig('heatmap_Q_r0_ifr.png', dpi=300)

fig = plt.figure(figsize=(6,5))
ax=fig.add_subplot(111)
im = ax.pcolormesh(R0, IFR*100, T(Reff), cmap='RdYlGn_r')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("$R_\\mathrm{eff}$",fontsize=16)
CS = ax.contour(R0,IFR*100, T(Reff), levels=[1.1,1.2,1.3,1.4,1.5,1.6,1.7], colors='k')
ax.clabel(CS, inline=1, fontsize=10,fmt='%1.1f')
ax.set_xlabel( "$R_0$", fontsize=16 )
ax.set_ylabel( "IFR (%)", fontsize=16)
plt.tight_layout()
plt.savefig('heatmap_Reff_r0_ifr.png', dpi=300)

fig = plt.figure(figsize=(6,5))
ax=fig.add_subplot(111)
im = ax.pcolormesh(R0, IFR*100, T(P), cmap='RdYlGn_r')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("$P$",fontsize=16)
CS = ax.contour(R0,IFR*100, T(P), levels=[0.0025,0.005,0.01,0.02,0.04,0.08], colors='k')
ax.clabel(CS, inline=1, fontsize=10,fmt='%1.3f')
ax.set_xlabel( "$R_0$", fontsize=16 )
ax.set_ylabel( "IFR (%)", fontsize=16)
plt.tight_layout()
plt.savefig('heatmap_P_r0_ifr.png', dpi=300)

fig = plt.figure(figsize=(6,5))
ax=fig.add_subplot(111)
cmap = cm.get_cmap('RdYlGn', 12)    # 11 discrete colors
im = ax.pcolormesh(R0, IFR*100, T(If)*100, cmap=cmap,vmin=10,vmax=70)
#CS = ax.contour(R0,IFR*100, T(If)*100, levels=[10,15,20,30,40], colors='k')
#ax.clabel(CS, inline=1, fontsize=10,fmt='%1.0f')
CS = ax.contour(R0,IFR*100, T(Q), levels=[0.3,0.4,0.5,0.6,0.7,0.8,0.9], colors='k',linestyles='-')
ax.clabel(CS, inline=1, fontsize=10,fmt='Q=%1.1f')
plt.plot( [R0.min(),R0.max()],[1.25,1.25], ls='--', lw=2, color='w')
plt.annotate( "Molenbergs et al. (2020)", (R0.mean()-0.33*(R0.max()-R0.mean()), 1.2), color='w', fontsize=13,fontstyle='italic' )
plt.plot( [R0_set], [IFR_set*100], marker='x', mew=4, ms=15, color='k',ls='None')
plt.annotate( "Best fit", (R0_set-1.0, IFR_set*100), color='k', fontsize=13,fontstyle='italic' )
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Est. accumulated infected (%)",fontsize=16)
ax.set_xlabel( "$R_0$", fontsize=16 )
ax.set_ylabel( "IFR (%)", fontsize=16)
plt.tight_layout()
plt.savefig('heatmap_If_r0_ifr.png', dpi=300)

fig = plt.figure(figsize=(6,5))
ax=fig.add_subplot(111)
im = ax.pcolormesh(R0, IFR*100, T(RSQi), cmap='RdYlGn_r',vmax=RSQi.max(), vmin=0.95*RSQi.max())
CS = ax.contour(R0,IFR*100, T(RSQi), levels=[0.974], colors='w')
ax.clabel(CS, inline=1, fontsize=10,fmt='%1.3f')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("$r^2_i$",fontsize=16)
ax.set_xlabel( "$R_0$", fontsize=16 )
ax.set_ylabel( "IFR (%)", fontsize=16)
plt.tight_layout()
plt.savefig('heatmap_rsqi_r0_ifr.png', dpi=300)

imax,jmax = np.unravel_index(np.argmax(RSQ), np.shape(RSQ))
#Figure of fit quality
fig = plt.figure(figsize=(6,5))
ax=fig.add_subplot(111)
im = ax.pcolormesh(R0, IFR*100, T(np.log10(1-RSQ)), cmap='RdYlGn_r')
#CS = ax.contour(R0,IFR*100, smooth_image(log10(1-RSQ),sigma=1 ), levels=[np.log10(1-0.997)], colors='k')
#ax.clabel(CS, inline=1, fontsize=10,fmt='%1.2f')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("$\\mathrm{log}_{10}(1-r^2)$",fontsize=16)
ax.set_xlabel( "$R_0$", fontsize=16 )
ax.set_ylabel( "IFR (%)", fontsize=16)
plt.tight_layout()
plt.savefig('heatmap_rsq_r0_ifr.png', dpi=300)


