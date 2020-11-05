import numpy as np
import matplotlib.pyplot as plt
import copy
import datetime
import pickle

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

th_rsq = 0.98

with open('heatmap_results.pickle','rb') as fob:
   data = pickle.load(fob)

#Unpack the data:
R0,IFR,rdict = data[0],data[1],data[2]

PICU = np.zeros( ( len(R0),len(IFR) ) )
RSQ = np.zeros( np.shape(PICU) )
Q = np.zeros(np.shape(PICU))
DICU = np.zeros(np.shape(PICU))

for i,R0i in enumerate(R0):
   for j,IFRi in enumerate(IFR):
      results = rdict[(i,j)]
      PICU[i,j] = results['icu_peak'] if results['rsq'] > th_rsq else np.nan
      RSQ[i,j] = results['rsq']
      Q[i,j] = results['Q'] if results['rsq'] > th_rsq else np.nan
      DICU[i,j] = results['d_peak'] if results['rsq'] > th_rsq else np.nan

PICU = np.ma.masked_where(np.isnan(PICU),PICU)
DICU = np.ma.masked_where(np.isnan(DICU),DICU)
Q = np.ma.masked_where(np.isnan(Q),Q)

#Figure of Peak ICU occupation
fig = plt.figure(figsize=(6,5))
ax=fig.add_subplot(111)
im = ax.pcolormesh(R0, IFR*100, T(PICU), cmap='RdYlGn_r')
CS = ax.contour(R0,IFR*100, T(PICU), levels=[1750,2000,2250,2500,2750,3000,3250,3500], colors='k')
ax.clabel(CS, inline=1, fontsize=10,fmt='%1.0f')
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
CS = ax.contour(R0,IFR*100, T(DICU/7.), levels=[1,2,3,4], colors='k')
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
ax.set_xlabel( "$R_0$", fontsize=16 )
ax.set_ylabel( "IFR (%)", fontsize=16)
plt.tight_layout()
plt.savefig('heatmap_Q_r0_ifr.png', dpi=300)

imax,jmax = np.unravel_index(np.argmax(RSQ), np.shape(RSQ))
#Figure of fit quality
fig = plt.figure(figsize=(6,5))
ax=fig.add_subplot(111)
im = ax.pcolormesh(R0, IFR*100, T(np.log10(1-RSQ)), cmap='RdYlGn_r', vmax=-1)
#CS = ax.contour(R0,IFR*100, smooth_image(log10(1-RSQ),sigma=1 ), levels=[np.log10(1-0.997)], colors='k')
#ax.clabel(CS, inline=1, fontsize=10,fmt='%1.2f')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("$\\mathrm{log}_{10}(1-r^2)$",fontsize=16)
ax.set_xlabel( "$R_0$", fontsize=16 )
ax.set_ylabel( "IFR (%)", fontsize=16)
plt.tight_layout()
plt.savefig('heatmap_rsq_r0_ifr.png', dpi=300)

