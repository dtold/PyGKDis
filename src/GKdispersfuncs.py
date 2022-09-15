"""
Initial development: Feb 2014 - May 2015
Authors: Daniel Told & Frank Jenko
"""

# Gyrokinetic dispersion relation solver for a hydrogen plasma
# in a homegeneous magnetic field, including phi, Apar and Bpar
# Ref.: Howes et al., ApJ 651, 590 (2006)

################################################################
####################### IMPORT MODULES #########################
################################################################
from __future__ import division,absolute_import,print_function,unicode_literals #for Python 2.7

from pylab import *
from scipy.special import wofz,i0,i1
from scipy.optimize import fsolve,root,minimize

from numpy import amin,zeros,matrix,tile,linspace,repeat,empty,log10,sqrt,exp,pi
import numpy as np
from numpy.linalg import det,eig
from matplotlib.pyplot import contourf,colorbar,show

################################################################
############### DEFINE Helper FUNCTIONS ########################
################################################################

#
# Gamma functions
#
def Gamma0(alpha):
    if alpha > alpha_c :
        return 1/(sqrt(2*pi*alpha))*(1+1/(8*alpha))
    else:
        return i0(alpha)*exp(-alpha)

def Gamma1(alpha):
    if alpha > alpha_c :
        return 1/(2*alpha*sqrt(2*pi*alpha))
    else:
        return (i0(alpha)-i1(alpha))*exp(-alpha)

def Gamma2(alpha):
    return 2*Gamma1(alpha)

#
# plasma dispersion function
#
def Z(zeta):
    return wofz(zeta)*1j*sqrt(pi)

#
# define terms A-E in Howes paper (Eqs.42-46)
#
def A(epsrat=1.):
    zeta_i = zeta; zeta_e = zeta_e_g
    alpha_i = alpha; alpha_e = alpha/mu_g/tau_g
    iterm = 1 + Gamma0(alpha_i) * zeta_i * Z(zeta_i)
    eterm = 1 + Gamma0(alpha_e) * zeta_e * Z(zeta_e)
    return iterm + epsrat * tau_g * eterm

def B(epsrat=1.):
    alpha_i = alpha; alpha_e = alpha/mu_g/tau_g
    iterm = 1 - Gamma0(alpha_i)
    eterm = 1 - Gamma0(alpha_e)
    return iterm + epsrat * tau_g * eterm

def C(epsrat=1.):
    zeta_i = zeta; zeta_e = zeta_e_g
    alpha_i = alpha; alpha_e = alpha/mu_g/tau_g
    iterm = Gamma1(alpha_i) * zeta_i * Z(zeta_i)
    eterm = Gamma1(alpha_e) * zeta_e * Z(zeta_e)
    return iterm - eterm / epsrat

def D(epsrat=1.):
    zeta_i = zeta; zeta_e = zeta_e_g
    alpha_i = alpha; alpha_e = alpha/mu_g/tau_g
    iterm = 2 * Gamma1(alpha_i) * zeta_i * Z(zeta_i)
    eterm = 2 * Gamma1(alpha_e) * zeta_e * Z(zeta_e)
    return iterm + eterm / tau_g / epsrat

def E():
    alpha_i = alpha; alpha_e = alpha/mu_g/tau_g
    iterm = Gamma1(alpha_i)
    eterm = Gamma1(alpha_e)
    return iterm - eterm

#
# Dispersion Relation Functionos
#

#no collision
def dispersion_relation_no_coll(omega):
    global zeta, mat, zeta_e_g
    zeta = omega / sqrt(beta_i_g)
    zeta_e_g=zeta/sqrt(mu_g/tau_g)
    vec1 = [A(), A()-B(), C()]
    vec2 = [A()-B(), A()-B()-alpha/omega**2, C()+E()]
    vec3 = [C(), C()+E(), D()-2/beta_i_g]
    mat = matrix([vec1, vec2, vec3])
    val=abs(det(mat))
    return abs(val)

#
# dispersion relation function with an interface as required by fsolv
#
def dr_call(x):
        #
        # function that calls the correct dispersion relation.
        #
    omega_c = complex(x[0], x[1])
    zeta = omega_c / sqrt(beta_i_g)
    out=dispersion_relation_no_coll(omega_c)
    return [out,out]

def DR_rf(x):
    return dr_call(x)


def DR_min(x):
    out=dr_call(x)
    return abs(np.dot(out,out))

def make_global(params):
    #
    # make parameters global
    #
    global kperp_g,mu_g,tau_g,beta_i_g
    kperp_g=params[0]
    mu_g=params[1]
    tau_g=params[2]
    beta_i_g=params[3]

def GKSolve(params,drtype_local="NoCo",start=[1.0,0]):

    ################################################################
    ###################### INPUT PARAMETERS ########################
    ################################################################
    #
    # make input parameters global for other functions
    #
    global start_g,drtype,kperp_c,alpha,alpha_c

    kperp_c = 20 #upper limit of kperp
    alpha=params[0]**2
    alpha_c=kperp_c**2

    start_g=start
    drtype=drtype_local

    make_global(params)

    ################################################################
    ################## SOLVE EIGENVALUE PROBLEM ####################
    ################################################################

    #
    # find a solution of the equation DR_rf = 0 given the starting estimate x0
    #

    solt=root(DR_rf, x0=start,tol=1e-9)
    if solt.get('success')==False:
        print("trying fsolve: ")
        x, infodict, ier, mesg = fsolve(DR_rf, x0=start, maxfev=0,
                                    full_output=1, xtol=1e-9, epsfcn=1e-6) #changed xtol from 1e-6
        nfev=infodict['nfev']

        if ier:
            print('fsolve success: ')

    else:
        print("root solver success: ")
        x=solt.get('x')
        ier=1
        mesg=solt.get('message')
        nfev=solt.get('nfev')

    if ier:
        print ("Converged after", nfev, "iterations")
        print ("Complex frequency:", (x[0], x[1]))
        out=dr_call(x)
        print ("Dispersion relation residual:", np.dot(out,out))

        omega_z = complex(x[0], x[1])
        eigvec=None #assume it is nothing, if we find it, then it will be changed.
        compeigvec=None #for complex components
        u, v = eig(mat)
        eps=2*amin(abs(u))
        move=abs(x[0]-start[0])+abs(x[1]-start[1]) #will be 0 if it doesn't move

        #
        # check to see if the solver moved and if the eigenvalue is close to 0
        #
        if (move<1e-9 or eps >= 1e-3):
            #
            # Try minimization if it isn't.
            #
            #
            # Not sure this actually helps
            #
            print('No near-zero eigenvalue found, trying minimization')
            solt=minimize(DR_min,x0=start,method='Nelder-Mead',options={'xtol':1e-9}) #method='Nelder-Mead' or method='BFGS'
            if solt.get('success')==True:
                print("minimize success: ")
                x=solt.get('x')
                ier=1
                mesg=solt.get('message')
                nfev=solt.get('nfev')
                omega_z = complex(x[0], x[1])
                dr_call(x) #to update mat
                u, v = eig(mat)
                eps=2*amin(abs(u))
                print ("New complex frequency:", (x[0], x[1]))
            else:
                ier=0
        #
        # Get the eigenvectors.
        #
        #print('Smallest eigenvalue',eps)
        phi=None
        bpar=None
        for i in range(3):
            if abs(u[i]) < eps and eps < 1e-4:
                print ("Smallest eigenvalue and corresponding eigenvector:")
                print (abs(u[i]))
                eigvec=zeros((3))
                compeigvec=np.array([0+1j*0,0+1j*0,0+1j*0])
                for j in range(3):
                    eigvec[j]=abs(v[j,i])   #magnitude
                    compeigvec[j]=v[j,i]
                    
                print("Phi/Apar: ", eigvec[0]/eigvec[1])
                print("Apar/Bpar: ",eigvec[1]/eigvec[2])

                #
                # Eigen vecs
                #
                phi=compeigvec[0]/compeigvec[1]
                bpar=compeigvec[2]/compeigvec[1]

                #
                # Filter out bad eigenvectors
                #
                if eigvec[2]==0:
                    phi=None
                    bpar=None
                elif eigvec[1]/eigvec[2] > 1e13: #haven't seen a correct ratio like this yet
                    phi=None
                    bpar=None

        return x[0],x[1],phi,bpar,1 #1 means good
    else:
        print (mesg)
        return 0,0,None,None,0 #0 means bad

def PlotDR2d(params,drtype_local,np=100,cntr=[0,-1],raxis=3.,iaxis=3.):
    global drtype,kperp_c,alpha,alpha_c

    kperp_c = 20 #upper limit of kperp
    alpha=params[0]**2
    alpha_c=kperp_c**2

    drtype=drtype_local

    make_global(params)

    xmin = cntr[0] - raxis/2; xmax = cntr[0] + raxis/2
    ymin = cntr[1] - iaxis/2; ymax = cntr[1] + iaxis/2
    arr1 = tile(linspace(xmin, xmax, np), np)
    arr2 = repeat(linspace(ymin, ymax, np), np)
    omega_out = (arr1 + 1j * arr2).reshape(np, np)
    omega = (arr1 + 1j * arr2)


    DR=empty((np*np),dtype=float)
    for i in range(len(omega)):
        out=dr_call([omega[i].real,omega[i].imag])
        DR[i] = abs(complex(out[0],out[1]))
    DRout = DR.reshape(np, np)

    return omega_out,DRout
