import sys
import os

#add source code directory to path
startdir=os.path.join(os.path.split(sys.argv[0])[0])
sys.path+=[os.path.join(startdir,'src')]
from GKdispersfuncs import GKSolve

#
# For finding solutions at a single point
#

kperp=1.0
mu=1836.
tau=1.
beta=1.

#guess for starting point in (omega, gamma)
start=[1.0,-.04]

##############################3
params=[kperp,mu,tau,beta]

#run the solver
GKSolve(params,start)
