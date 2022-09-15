import sys
import os

#add source code directory to path
startdir=os.path.join(os.path.split(sys.argv[0])[0])
sys.path+=[os.path.join(startdir,'src')]

from GKdispersfuncs import PlotDR2d
from matplotlib.pyplot import plot,contourf,colorbar,show,imshow
from numpy import log10, amin, amax

#
# parameters
#
kperp=1.0
beta=1.
tau=1.
mu=1836.

np=200       #number of points in plot (np**2 points total)
raxis=5.     #length of real axis
iaxis=5.     #legnth of imag axis
eps=0.01     #axis offset from 0,0
cntr=[raxis/2+eps,-iaxis/2-eps] #center of graph

params=[kperp,mu,tau,beta]

nlev=20 #number of levels in contour plot

#
# this function gets data to plot
#
xy,z=PlotDR2d(params,"NoCo",np,cntr,raxis,iaxis)

#plot the data

###contour plot and imshow have inverted y axes – reason for keeping both is absence of interpolation in imshow
#contourf(xy.real,xy.imag,log10(z),nlev,cmap='gist_heat_r')
imshow(log10(z),cmap='gist_heat_r',extent=[amin(xy.real),amax(xy.real),amax(xy.imag),amin(xy.imag)],interpolation='none')
colorbar()
show()
