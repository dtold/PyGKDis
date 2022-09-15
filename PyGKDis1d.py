from __future__ import division, print_function
import sys
import os

#add source code directory to path
startdir=os.path.join(os.path.split(sys.argv[0])[0])
sys.path+=[os.path.join(startdir,'src')]

import numpy as np
from GKGeneralScanFuncs import GK_Scan_scanvar,Plot_Scan
from sys import stderr
#
# Pick your starting values
#
#Normal start
#starts for kperp=beta=tau=1, mu=1836
#will track alfven waves and first few modes (non-entropy)
#with the parameters tau=1,beta=1,mu=1836
starts=[
        [1.28,-0.48],
        [1.13,-1.0],
        [1.69,-1.63],
        [2.20,-1.97],
        [2.48,-2.36],
        [2.85,-2.64],
        [3.07,-2.93]]

#
# Label the starting value modes (for plotting)
#
labels=['Alfven wave', 'first damped', 'second damped', 'third damped', 'fourth damped', 'fifth damped',
                'sixth damped']

precis=500*np.ones(len(starts),dtype=int) #number of points along the scan range
#
# can specify individual precisions by precis[?]=200
#

####################
#################### pick variable to scan
####################
scanname='kperp' #options are 'mu', 'kperp', 'tau','beta'

# label to appear on x-axis of plots
scanlabel=r'$k_{\perp}\rho_i$'
scanrange=[0.01,1,100]  #scanrange is given as a list of three numbers: [lower, mid, upper]
                        #The full range is divided in two, so the chosen starting point can be
                        #anywhere in the range.
                        #"mid" is the starting point, and "lower"/"upper" define the boundaries
                        #of the scan range.
                        #All scans are done with logarithmic spacing.

drtype = 'NoCo' 
###################
################### Put in parameters at starting point.
###################
kperp0=1.
mu0=1836.
tau0=1.
beta0=1.
params=[kperp0,mu0,tau0,beta0] #order is kperp,mu,tau,beta
##########################
##########################
XARRAYS=[]
OMEGAS=[]
GAMMAS=[]
PHIS=[]
BPARS=[]
for i in range(len(starts)):
    print('at mode', i, file=stderr) #keep track of what mode the solve is doing

    #
    # This function actually does the scanning
    #
    OUT=GK_Scan_scanvar(params,drtype,start=starts[i],scanvar=scanname,precis=precis[i],scan_range=(np.log10(scanrange[1]),np.log10(scanrange[2])))
    #
    #

    #
    # Get values from scan
    #
    XTEMP=OUT[0]
    OTEMP=OUT[1]
    GTEMP=OUT[2]
    PTEMP=OUT[3]
    BTEMP=OUT[4]

    #
    # This next scan goes in the other direction so you can scan over parameter space with just
    # knowing a single point
    #
    OUT2=GK_Scan_scanvar(params,drtype,start=starts[i],scanvar=scanname,precis=precis[i],scan_range=(np.log10(scanrange[1]),np.log10(scanrange[0])))

    #
    # Combine values from both scans
    #
    XTEMP=np.append(OUT[0][::-1],OUT2[0][1:])
    OTEMP=np.append(OUT[1][::-1],OUT2[1][1:])
    GTEMP=np.append(OUT[2][::-1],OUT2[2][1:])
    PTEMP=np.append(OUT[3][::-1],OUT2[3][1:])
    BTEMP=np.append(OUT[4][::-1],OUT2[4][1:])

    #
    # Put them in a large array for plotting
    #
    XARRAYS.append(XTEMP.tolist())
    OMEGAS.append(OTEMP.tolist())
    GAMMAS.append(GTEMP.tolist())
    PHIS.append(PTEMP.tolist())
    BPARS.append(BTEMP.tolist())

    #
    # print the data if you want to save for later
    #
    print('for the',i,'th mode')
    print('XARRAY=',XTEMP.__repr__())
    print('OMEGA=',OTEMP.__repr__())
    print('GAMMA=',GTEMP.__repr__())
    print('PHI=',PTEMP.__repr__())
    print('BPAR=',BTEMP.__repr__())

#
# print all the data in array form
#
print('for the scan')
print('XARRAYS=',XARRAYS.__repr__())
print('OMEGAS=',OMEGAS.__repr__())
print('GAMMAS=',GAMMAS.__repr__())
print('PHIS=',PHIS.__repr__())
print('BPARS=',BPARS.__repr__())

#
# Plot the functions
#
Plot_Scan(scanlabel,XARRAYS,OMEGAS,GAMMAS,PHIS,BPARS,labels)
