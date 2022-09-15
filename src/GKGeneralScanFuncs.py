from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
from numpy import log10
from sys import  stderr
from GKdispersfuncs import GKSolve


def GK_Scan_scanvar(params,drtype="NoCo",start=[1.0,0.0],scanvar='kperp',precis=1000,scan_range=(-1,2),first_point=[]):
    # This function allows the roots to be found ranging across XARRAY
    # There are two main techniques for finding roots. Scanning around
    # many different starting positions and backtracking.
    XARRAY=np.logspace(scan_range[0],scan_range[1],num=precis)
    OMEGA=np.zeros(precis)
    GAMMA=np.zeros(precis)
    PHI=np.zeros(precis,np.complex_) #normalized to APAR
    BPAR=np.zeros(precis,np.complex_) #normalized to APAR
    i=0
    if len(first_point)!=0:
        #The first point is treated differently, so it is safest
        #to pass along this information if available like in 2d Scan.
        OMEGA[0]=first_point[0]
        GAMMA[0]=first_point[1]
        PHI[0]=first_point[2]
        BPAR[0]=first_point[3]
        i=1
    while i<len(XARRAY): #Due to backtracking XARRAY may expand.
        print("HERE",i)
        scan=call_GK_get_root(i,params,drtype,start,scanvar,XARRAY,OMEGA,GAMMA,PHI,BPAR)
        print("[omega,gamma,phi,bpar,err]: ",scan)
        omega,gamma,phi,bpar,err=scan[0],scan[1],scan[2],scan[3],scan[4]
        if err==1: # no error
            OMEGA[i]=omega
            GAMMA[i]=-gamma
            PHI[i]=phi
            BPAR[i]=bpar
        else:
            #try splitting the difference of scanvar
            # hopefully backtracking will let us find the root.
            sv0=XARRAY[i-1]
            sv1=XARRAY[i]
            savesv=sv1
            count=0
            while (err==0 and count < 25):
                #divide by 2
                XARRAY[i]=(sv1+sv0)/2
                sv1=XARRAY[i] #the value we just tried is in XARRAY[i]
                print("trying backtracking scanvar=",XARRAY[i])
                omega,gamma,phi,bpar,err=call_GK_get_root(i,params,drtype,start,scanvar,XARRAY,OMEGA,GAMMA,PHI,BPAR)
                print([omega,gamma,phi,bpar,err])
                count+=1
                print("Count: ", count,sv0,sv1,XARRAY[i])
            if err==1:# no error
                print("backtracking worked")
                #put in found values
                OMEGA[i]=omega
                GAMMA[i]=-gamma
                PHI[i]=phi
                BPAR[i]=bpar
                #expand arrays and adjust XARRAY
                OMEGA=np.append(OMEGA,0)
                GAMMA=np.append(GAMMA,0)
                PHI=np.append(PHI,0)
                BPAR=np.append(BPAR,0)
                #### XARRAY ####
                # a little confusing but we are just missing the i+1th entry
                # which should be the old ith entry.
                XARRAY=np.append(XARRAY[0:i+1],np.append(np.array([savesv]),XARRAY[i+1:]))
                print( len(XARRAY) )
            else:
                print("Error no root found:", XARRAY[i])
                print("ERROR NO ROOT ENDING SCAN AT SCANVAR=",XARRAY[i],file=stderr)
                return XARRAY[:i],OMEGA[:i],GAMMA[:i],PHI[:i],BPAR[:i]
        i+=1
    return XARRAY, OMEGA, GAMMA,PHI,BPAR

def call_GK_get_root(index=0,params=[],drtype='NoCo',start=[1.0,0.0],scanvar='kperp',XARRAY=[],OMEGA=[],GAMMA=[],PHI=[],BPAR=[]):
    #This function lets us specify which variable we are scanning so the solver can know.
    if scanvar=='kperp':
        params[0]=XARRAY[index]
    elif scanvar=='beta':
        params[3]=XARRAY[index]
    elif scanvar=='tau':
        params[2]=XARRAY[index]
    elif scanvar=='mu':
        params[1]=XARRAY[index]
    else:
        print("Pick a valid scanvar.",file=stderr)
    return GK_get_root(index,params,drtype,start,XARRAY,OMEGA,GAMMA,PHI,BPAR)

def next_start(X, F, G, i, j,scandirec=0):
    #this lets you scan around the previous position to hopefully
    #find the roots new position. Scandirec lets you do a scan along
    #the line gamma=omega*scandirec with the origin at the previous
    #found root spot.
    k=scandirec*j
    gshift=k/10**3
    if gshift>G[i]: #don't let gamma become positive
        gval=G[i]
    else:
        gval=G[i]-k/10**3
    return [F[i]+j/10**3,-gval]

def GK_get_root(index=0,params=[],drtype='NoCo',start=[1.0,0.0],XARRAY=[],OMEGA=[],GAMMA=[],PHI=[],BPAR=[]):
    # finds the root by using different starting postions
    # as well as checking to make sure the root it converged to
    # is the correct one.
    if index==0:
        omega,gamma,phi,bpar,err=GKSolve(params,drtype,start)
        while phi is None:
            start[0]+=start[0]/10**5 #might not work great.
            omega,gamma,phi,bpar,err=GKSolve(params,drtype,start)
        return [omega,gamma,phi,bpar,err]

    def Scan(jstart,jstep,jmax,scandirec=0):
        #
        # Scan around the starting point looking for a good solution
        #
        j=jstart
        while (j<jmax):
            start=next_start(XARRAY,OMEGA,GAMMA,index-1,j,scandirec)

            print("params,drtype,start=",params,drtype,start)
            #
            # This function gets the root
            #
            omega,gamma,phi,bpar,err=GKSolve(params,drtype,start)
            if err==1 and (phi is not None): #no obvious error.
                if abs(omega)<1e3 and abs(gamma)<1e3 and gamma<0:
                    #to prevent obviously wrong solutions.
                    if index==0 or abs(omega-OMEGA[index-1])>1e-9 or abs(gamma+GAMMA[index-1])>1e-9:
                        #
                        # make sure solver has moved
                        #
                        print ('omega,gamma,phi,bpar=',omega,gamma,phi,bpar)
                        print ("ecart =", np.sqrt((omega-start[0])**2+(gamma-start[1])**2))
                        print ("start=",start)
                        limit=.05
                        #phi/apar continuity:
                        change=abs(log10(abs(phi*abs(omega+1j*gamma)))-
                                log10(abs(PHI[index-1]*abs(OMEGA[index-1]-1j*GAMMA[index-1]))))
                        #apar/bpar continuity:
                        change2=abs(log10(abs(bpar*(omega+1j*gamma)))-
                                        log10(abs(BPAR[index-1]*(OMEGA[index-1]-1j*GAMMA[index-1]))))
                        if (abs(log10(abs(omega))-log10(abs(start[0])))<limit and (abs(log10(-gamma)-log10(-start[1]))<limit
                                                                        or start[1]<1e-6) and
                                                                        (change < limit and change2 < limit)):
                            #
                            # make sure solver hasn't jumped
                            #
                            # allow gamma, when really small, some more leeway.
                            print(change, change2)
                            print(omega,start[0])
                            print(gamma,start[1])
                            print("found good root\n")
                            return [omega,gamma,phi,bpar,err]
                        else:
                            #last chance is if the mode is becoming an entropy mode
                            #find approximate slope of loglog graph
                            #it should be very negative when becoming entropy mode
                            deltax=log10(XARRAY[index-1])-log10(XARRAY[index-2])
                            deltay=log10(OMEGA[index-1])-log10(OMEGA[index-2])
                            slope=deltay/deltax
                            gammachange=abs(log10(-gamma)-log10(-start[1]))
                            if (omega < 1e-7 and (slope < -5. or OMEGA[index-1]<1e-7)
                                            and gammachange < limit):
                                #entropy mode or something like it, so call it a good root
                                print ('found good ENTROPY-LIKE root')
                                return [omega,gamma,phi,bpar,err]
                            print("bad root: continuity issue")
                            j+=jstep
                    else:
                        print("too close to start")
                        j+=jstep
                else:
                    print ("too far away")
                    j+=jstep
            else:
                print("eigs were None or solver error")
                j+=jstep
        return None

    # Run many scans to increase likelihood of finding root.
    scan_1=Scan(0.0,1.0,1.0,scandirec=0.)
    if scan_1 is None:
        scandirec=-1.
        while (scan_1 is None and scandirec<1.5):
            scan_1=Scan(-0.1,.04,0.1,scandirec)
            scandirec+=1.
        if scan_1 is None:
            scandirec=-1.
            while (scan_1 is None and scandirec<1.5):
                scan_1=Scan(-3.0,.5,3.0,scandirec)
                scandirec+=1.
            if scan_1 is None:
                return 0,0,None,None,0
    #
    # Might want to consider doing fewer scans above.
    #
    return scan_1[0],scan_1[1],scan_1[2],scan_1[3],scan_1[4]

def Plot_Scan(scanlabel,XARRAYS,OMEGAS,GAMMAS,PHIS,BPARS,LABELS):
    #here we plot everything
    #These are all multidimensional arrays
    #
    # First we make axes labels
    #
    plt.figure(1) #omega vs. scanvar
    plt.xlabel(scanlabel)
    plt.ylabel(r'$\omega/(k_{\parallel}v_A)$')
    plt.figure(2) #gamma vs. scanvar
    plt.xlabel(scanlabel)
    plt.ylabel(r'$-\gamma/(k_{\parallel}v_A)$')
    plt.figure(3) #GAMMA/OMEGA vs. scanvar
    plt.xlabel(scanlabel)
    plt.ylabel(r'$-\gamma/\omega$')
    plt.figure(4) #Phi/Apar vs. scanvar
    plt.xlabel(scanlabel)
    plt.ylabel(r'$\phi/A_{\parallel}$')
    plt.figure(5) #Bpar/Apar vs. scanvar
    plt.xlabel(scanlabel)
    plt.ylabel(r'$B_{\parallel}/A_{\parallel}$')
    plt.figure(6) #Bpar/Phi vs. scanvar
    plt.xlabel(scanlabel)
    plt.ylabel(r'$B_{\parallel}/\phi$')
    #
    # Now we plot the data
    #
    for i in range(0,len(LABELS)):
        Xtemp=np.asarray(XARRAYS[i])
        Otemp=np.asarray(OMEGAS[i])
        Gtemp=np.asarray(GAMMAS[i])
        Ptemp=np.asarray(PHIS[i])
        Btemp=np.asarray(BPARS[i])
        plt.figure(1)
        plt.loglog(Xtemp,Otemp,label=LABELS[i])
        plt.figure(2)
        plt.loglog(Xtemp,Gtemp,label=LABELS[i])
        plt.figure(3)
        plt.loglog(Xtemp,Gtemp/Otemp,label=LABELS[i])
        RAT1=abs(Ptemp*(Otemp+1j*Gtemp))
        RAT2=abs(Btemp*(Otemp+1j*Gtemp))
        plt.figure(4)
        plt.loglog(Xtemp,RAT1,label=LABELS[i])
        plt.figure(5)
        plt.loglog(Xtemp,RAT2,label=LABELS[i])
        plt.figure(6)
        plt.loglog(Xtemp,RAT2/RAT1,label=LABELS[i])
    #
    # make legends
    #
    plt.figure(1)
    plt.legend(loc='upper left', prop={'size':7})
    plt.figure(2)
    plt.legend(loc='upper left', prop={'size':7})
    plt.figure(3)
    plt.legend(loc='upper left', prop={'size':7})
    plt.figure(4)
    plt.legend(loc='upper left', prop={'size':7})
    plt.figure(5)
    plt.legend(loc='upper left', prop={'size':7})
    plt.figure(6)
    plt.legend(loc='upper left', prop={'size':7})
    #
    # show graphs
    #
    plt.show()
