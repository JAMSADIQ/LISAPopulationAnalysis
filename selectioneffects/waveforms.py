""" This script computes the amplitude of the phenomC waveform (no phase computation)
It is adapted from Emma Robinson's C-code.
Reference to the paper: http://arxiv.org/abs/1005.3306 unless otherwise stated
"""

import sys
import os
import re
#import getopt
import numpy as np
import math

from  constants import *

from numpy.fft import *


from math import pi, sin, cos, sqrt, pow



"""
******************************************************************************
* phenomenological parameters                                                *
* from Table II in http://arxiv.org/abs/1005.3306                            *
******************************************************************************
"""

BBHNEWPHENOMCOEFFSH_ALPHA1_ZETA01 = -2.417e-03
BBHNEWPHENOMCOEFFSH_ALPHA1_ZETA02 = -1.093e-03
BBHNEWPHENOMCOEFFSH_ALPHA1_ZETA11 = -1.917e-02
BBHNEWPHENOMCOEFFSH_ALPHA1_ZETA10 =  7.267e-02
BBHNEWPHENOMCOEFFSH_ALPHA1_ZETA20 = -2.504e-01

BBHNEWPHENOMCOEFFSH_ALPHA2_ZETA01 = 5.962e-01
BBHNEWPHENOMCOEFFSH_ALPHA2_ZETA02 = -5.600e-02
BBHNEWPHENOMCOEFFSH_ALPHA2_ZETA11 = 1.520e-01
BBHNEWPHENOMCOEFFSH_ALPHA2_ZETA10  = -2.970e+00
BBHNEWPHENOMCOEFFSH_ALPHA2_ZETA20  = 1.312e+01

BBHNEWPHENOMCOEFFSH_ALPHA3_ZETA01 = -3.283e+01
BBHNEWPHENOMCOEFFSH_ALPHA3_ZETA02 =  8.859e+00
BBHNEWPHENOMCOEFFSH_ALPHA3_ZETA11 =  2.931e+01
BBHNEWPHENOMCOEFFSH_ALPHA3_ZETA10 =  7.954e+01
BBHNEWPHENOMCOEFFSH_ALPHA3_ZETA20 = -4.349e+02

BBHNEWPHENOMCOEFFSH_ALPHA4_ZETA01 =  1.619e+02
BBHNEWPHENOMCOEFFSH_ALPHA4_ZETA02 = -4.702e+01
BBHNEWPHENOMCOEFFSH_ALPHA4_ZETA11 = -1.751e+02
BBHNEWPHENOMCOEFFSH_ALPHA4_ZETA10 = -3.225e+02
BBHNEWPHENOMCOEFFSH_ALPHA4_ZETA20  = 1.587e+03

BBHNEWPHENOMCOEFFSH_ALPHA5_ZETA01 = -6.320e+02
BBHNEWPHENOMCOEFFSH_ALPHA5_ZETA02  = 2.463e+02
BBHNEWPHENOMCOEFFSH_ALPHA5_ZETA11  = 1.048e+03
BBHNEWPHENOMCOEFFSH_ALPHA5_ZETA10  = 3.355e+02
BBHNEWPHENOMCOEFFSH_ALPHA5_ZETA20 = -5.115e+03

BBHNEWPHENOMCOEFFSH_ALPHA6_ZETA01 = -4.809e+01
BBHNEWPHENOMCOEFFSH_ALPHA6_ZETA02 = -3.643e+02
BBHNEWPHENOMCOEFFSH_ALPHA6_ZETA11 = -5.215e+02
BBHNEWPHENOMCOEFFSH_ALPHA6_ZETA10  = 1.870e+03
BBHNEWPHENOMCOEFFSH_ALPHA6_ZETA20  = 7.354e+02

BBHNEWPHENOMCOEFFSH_GAMMA1_ZETA01  = 4.149e+00
BBHNEWPHENOMCOEFFSH_GAMMA1_ZETA02 = -4.070e+00
BBHNEWPHENOMCOEFFSH_GAMMA1_ZETA11 = -8.752e+01
BBHNEWPHENOMCOEFFSH_GAMMA1_ZETA10 = -4.897e+01
BBHNEWPHENOMCOEFFSH_GAMMA1_ZETA20  = 6.665e+02

BBHNEWPHENOMCOEFFSH_DELTA1_ZETA01 = -5.472e-02
BBHNEWPHENOMCOEFFSH_DELTA1_ZETA02  =  2.094e-02
BBHNEWPHENOMCOEFFSH_DELTA1_ZETA11  = 3.554e-01
BBHNEWPHENOMCOEFFSH_DELTA1_ZETA10  = 1.151e-01
BBHNEWPHENOMCOEFFSH_DELTA1_ZETA20  = 9.640e-01

BBHNEWPHENOMCOEFFSH_DELTA2_ZETA01 = -1.235e+00
BBHNEWPHENOMCOEFFSH_DELTA2_ZETA02  = 3.423e-01
BBHNEWPHENOMCOEFFSH_DELTA2_ZETA11  = 6.062e+00
BBHNEWPHENOMCOEFFSH_DELTA2_ZETA10  = 5.949e+00
BBHNEWPHENOMCOEFFSH_DELTA2_ZETA20 = -1.069e+01

"""
******************************************************************************
* params for the tanh-window functions                                       *
* Eq. 5.8 in http://arxiv.org/abs/1005.3306                                  *
******************************************************************************
"""
BBHNEWPHENOMCOEFFSH_D_A = 0.015      # Width of window for amplitude fns
BBHNEWPHENOMCOEFFSH_D_P = 0.005      # Width of window for phase fns
BBHNEWPHENOMCOEFFSH_F0_COEFF = 0.98  # transition freq for amplitude fns
BBHNEWPHENOMCOEFFSH_F1_COEFF = 0.1   # lower transn freq for phase fns
BBHNEWPHENOMCOEFFSH_F2_COEFF = 1.0   # upper transn freq for phase fns


def AmpPhenomC(f, eta, chi, Mtot):
    # we assume chi1 = chi2 = chi, following the paper
    chi1 = chi;
    chi2 = chi;

    finalSpin = AEIFinalSpin(eta,chi);

    [alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, gamma1, delta1, delta2] = calcPhenParams(eta , chi );

    # fQNM frequency in units of M
    fRD = fQNM( math.fabs(finalSpin) );
    # Q: quality factor of ringdown
    Q = Qual( math.fabs(finalSpin) );

    # params for the PN part of the amplitude Eq A5*/

    #print "al1 = %f, al2 = %f, gam1 = %f, del2 = %f " % (alpha1, alpha2, gamma1, delta2)
    [A0, A1, A2, A3, A4, A5re, A5im, A6re, A6im] = calcPNParams( eta , chi1 , chi2 )
    PNpars = [A0, A1, A2, A3, A4, A5re, A5im, A6re, A6im]
    #print "A0 = %f, A1 = %f, A5re = %f, A5im = %f" % (A0, A1, A5re, A5im)

    [tp_a0, tp_a1, tp_a2, tp_a3, tp_a4, tp_a5, tp_a6, tp_a7] = calcTaylorParams(eta , chi , chi1 , chi2)
    TaylPars =  [tp_a0, tp_a1, tp_a2, tp_a3, tp_a4, tp_a5, tp_a6, tp_a7]

    #n = len(f)
    Ampl =  0. #np.zeros(n)

    freq=f * Mtot * mtsun;

    if (freq <= 0.15):
              x = pow( freq * pi , 2.0/3.0 );
              # PN amplitude from Eq 3.10
              AmpPN = AmpPNFn ( x , eta, PNpars, TaylPars)
              #print "AmpPN = ", AmpPN
              # pre-merger contribution Eq 5.11 */
              AmpPM = AmpPN + ( gamma1 * pow( freq , 5.0/6.0 ) );
              # ringdown contribution */
              L = LorentzianFn ( freq, fRD, fRD * delta2 / Q );
              # to correct for different defs of Lorentzian */
              L = L* 2.0 * pi * fRD * delta2 / Q;
              # ringdown amplitude Eq 5.12 */
              AmpRD = delta1 * L * pow( freq , -7.0/6.0 );

              # window functions Eq 5.8*/
              wPlus = TanhWindowPlus( freq , BBHNEWPHENOMCOEFFSH_F0_COEFF * fRD , BBHNEWPHENOMCOEFFSH_D_A );
              wMinus = TanhWindowMinus( freq , BBHNEWPHENOMCOEFFSH_F0_COEFF * fRD , BBHNEWPHENOMCOEFFSH_D_A );

              # phenom amplitude Eq 5.13 */
              Ampl = ( (AmpPM*wMinus) + (AmpRD*wPlus) ) ;
              #sys.exit(0)



    return(Ampl)



def LorentzianFn ( freq, fRing, sigma):

  out = sigma / (2. * pi * ((freq - fRing)*(freq - fRing) + sigma*sigma / 4.0));
  return(out);



def AmpPNFn (x, eta, PNpars, TaylPars):

  Amp22 = [0., 0.];


  Amp0 = sqrt( 2.0 * pi / ( 3.0 * sqrt(x) ) );
  #print "A0 = %f, A1 = %f, A5re = %f, A5im = %f" % (A0, A1, A5re, A5im)


  Amp22 = PNAmplitude22 (  x , PNpars );
  # real part
  Amp22[0] = Amp22[0] * 8.0*eta*x*sqrt(pi/5.0);
  # imag part
  Amp22[1] = Amp22[1] * 8.0*eta*x*sqrt(pi/5.0);

  XdotT4 = XdotT4Fn ( x , eta , TaylPars );

  fractionre=Amp22[0]/sqrt(XdotT4);
  fractionim=Amp22[1]/sqrt(XdotT4);

  Amp = Amp0 * sqrt((fractionre*fractionre)+(fractionim*fractionim));

  return(Amp);


def XdotT4Fn ( x , eta , tparams ):
   [p_a0, p_a1, p_a2, p_a3, p_a4, p_a5, p_a6, p_a7] = tparams
   x2=x*x;
   x3=x2*x;
   x52=sqrt(x2*x3);
   x72=x52*x;
   Amp0 = 64.0 * eta * pow(x, 5.0) / 5.0;

   Xdot = p_a0 + ( p_a1 * sqrt(x) ) + ( p_a2 * x ) + ( p_a3 * sqrt(x3) ) + ( p_a4 * x2 ) + \
        ( p_a5 * x52 ) + ( ( p_a6 - (856.0*math.log(16.0*x)/105.0) ) * x3 ) + ( p_a7 * x72 );

   Xdot = Xdot * Amp0;

   return(Xdot);



def PNAmplitude22 (x , params ):

  Amp = [0.0, 0.0]
  x2=x*x;
  x3=x2*x;
  x52=sqrt(x2*x3);
  [A0, A1, A2, A3, A4, A5re, A5im, A6re, A6im] = params
  Amp[0] = A0 + ( A1 * sqrt(x) ) + ( A2 * x ) + ( A3 * sqrt(x3) ) +\
        ( A4 * x2 ) + ( A5re * x52 ) + ( (A6re - (428.0*math.log(16.0*x)/105.0))* x3 );
  Amp[1] = ( A5im * x52 ) + ( A6im* x3 );
  #print "A0 = %f, A1 = %f, A5re = %f, A5im = %f" % (A0, A1, A5re, A5im)

  return(Amp)



def calcTaylorParams(eta , chi , chi1 , chi2 ):
    etasq = eta*eta;
    etacub=etasq*eta;
    chisq = chi*chi;
    chicub = chi*chisq;
    chi12 = chi1*chi2;
    etachi = eta*chi;
    pisq = pi*pi;

    tp_a0 = 1.0;
    tp_a1 = 0.0;

    tp_a2 = (-743.0/336.0) - (11.0*eta/4.0);

    tp_a3 = (4.0*pi) -  (113.0*chi/12.0) +  (19.0*eta*(chi1+chi2)/6.0);

    tp_a4 = (34103.0/18144.0) +  (5.0*chisq) + (eta*((13661.0/2016.0)-(chi12/8.0))) +\
         (59.0*etasq/18.0);

    tp_a5 = (-1.0*pi*((4159.0/672.0)+(189.0*eta/8.0))) -  (chi*((31571.0/1008.0)-(1165.0*eta/24.0))) +\
           ((chi1+chi2)*((21863.0*eta/1008.0)-(79.0*etasq/6.0))) - (3.0*chicub/4.0) + \
         (9.0*etachi*chi12/4.0);

    # a6 does not include the log(16x) part
    tp_a6 = (16447322263.0/139708800.0) - (1712.0*eulergamma/105.0) + (16.0*pisq/3.0) +\
            (eta*((451.0*pisq/48.0) -(56198689.0/217728.0))) + (541.0*etasq/896.0) -\
            (5605.0*etacub/2592.0) - (80.0*pi*chi/3.0) + (((20.0*pi/3.0)-(1135.0*chi/36.0))*eta*(chi1+chi2)) +\
            (((64153.0/1008.0)-(457.0*eta/36.0))*chisq) - (((787.0*eta/144.0)-(3037.0*etasq/144.0))*chi12);

    tp_a7 = (-1.0*pi * ( (4415.0/4032.0) - (358675.0*eta/6048.0) - (91495.0*etasq/1512.0) ) ) -\
            (chi*( (2529407.0/27216.0) - (845827.0*eta/6048.0) + (41551.0*etasq/864.0) )) +\
            ((chi1+chi2)*( (1580239.0*eta/54432.0) - (451597.0*etasq/6048.0) + (2045.0*etacub/432.0) +\
            (107.0*etachi*chi/6.0) - (5.0*etasq*chi12/24.0) )) +  (12.0*pi*chisq) - \
            (chicub*( (1505.0/24.0) + (eta/8.0) )) + (chi*chi12*( (101.0*eta/24.0) + (3.0*etasq/8.0) ));

    return(tp_a0, tp_a1, tp_a2, tp_a3, tp_a4, tp_a5, tp_a6, tp_a7)



def calcPNParams(eta , chi1 , chi2 ):

    chi12 = chi1*chi2;
    etasq = eta*eta;
    etacub = eta*etasq;
    pisq = pi*pi;

    A0 = 1.0;
    A1 = 0.0;
    A2 = (55.0*eta/42.0) - (107.0/42.0);
    A3 = (2.0*pi) + (-2.0*sqrt(1.0-(4.0*eta))*(chi1-chi2)/3.0) - \
         (2.0*(1.-eta)*(chi1+chi2)/3.0);
    A4 = (-2173.0/1512.0) - (eta*((1069.0/216.0)-(2.0*chi12))) + (etasq*2047.0/1512.0);
    A5re = (-107.0*pi/21.0) + (eta*(34.0*pi/21.0));
    A5im = -24.0 * eta;
    # A6 doesn't include the log(16x) part
    A6re = (27027409.0/646800.0) -  (856.0*eulergamma/105.0) +\
            (2.0*pisq/3.0) +  (eta*((41.0*pisq/96.0)-(278185.0/33264.0))) - \
            (etasq*20261.0/2772.0) + (etacub*114635.0/99792.0);
    A6im = (428.0*pi/105.0);
    return(A0, A1, A2, A3, A4, A5re, A5im, A6re, A6im)



def AEIFinalSpin(eta,chi):
  a = chi;
  s4 = -0.129;
  s5 = -0.384;
  t0 = -2.686;
  t2 = -3.454;
  t3 = 2.353;
  etasq = eta * eta;

  aFin = a + (s4 * a * a * eta) + (s5 * a * etasq) + (t0 * a * eta) + \
         (2.0 * sqrt(3.0) * eta) + (t2 * etasq) + \
         (t3 * etasq * eta);

  if (aFin>1.0) :
    aFin = 0.998

  return(aFin)


def fQNM( a ):
   f1=1.5251;
   f2=-1.1568;
   f3=0.1292;

   out = ( f1 + (f2*pow((1.0-a),f3)) )/(2.0 * pi);
   return(out);

def Qual( a ):

    q1 = 0.7;
    q2 = 1.4187;
    q3 = -0.499;

    out = q1 + ( q2 * pow( (1.0-a) , q3 ) );

    return(out);




def calcPhenParams( eta , chi ):
    etasq = eta*eta;
    etachi = eta*chi;
    chisq = chi*chi;

    alpha1 = BBHNEWPHENOMCOEFFSH_ALPHA1_ZETA01 * chi + \
             BBHNEWPHENOMCOEFFSH_ALPHA1_ZETA02 * chisq +\
             BBHNEWPHENOMCOEFFSH_ALPHA1_ZETA11 * etachi +\
             BBHNEWPHENOMCOEFFSH_ALPHA1_ZETA10 * eta + \
             BBHNEWPHENOMCOEFFSH_ALPHA1_ZETA20 * etasq;

    alpha2 = BBHNEWPHENOMCOEFFSH_ALPHA2_ZETA01 * chi + \
             BBHNEWPHENOMCOEFFSH_ALPHA2_ZETA02 * chisq +\
             BBHNEWPHENOMCOEFFSH_ALPHA2_ZETA11 * etachi +\
             BBHNEWPHENOMCOEFFSH_ALPHA2_ZETA10 * eta + \
             BBHNEWPHENOMCOEFFSH_ALPHA2_ZETA20 * etasq;

    alpha3 = BBHNEWPHENOMCOEFFSH_ALPHA3_ZETA01 * chi + \
             BBHNEWPHENOMCOEFFSH_ALPHA3_ZETA02 * chisq +\
             BBHNEWPHENOMCOEFFSH_ALPHA3_ZETA11 * etachi +\
             BBHNEWPHENOMCOEFFSH_ALPHA3_ZETA10 * eta + \
             BBHNEWPHENOMCOEFFSH_ALPHA3_ZETA20 * etasq;

    alpha4 = BBHNEWPHENOMCOEFFSH_ALPHA4_ZETA01 * chi + \
             BBHNEWPHENOMCOEFFSH_ALPHA4_ZETA02 * chisq +\
             BBHNEWPHENOMCOEFFSH_ALPHA4_ZETA11 * etachi +\
             BBHNEWPHENOMCOEFFSH_ALPHA4_ZETA10 * eta + \
             BBHNEWPHENOMCOEFFSH_ALPHA4_ZETA20 * etasq;

    alpha5 = BBHNEWPHENOMCOEFFSH_ALPHA5_ZETA01 * chi + \
             BBHNEWPHENOMCOEFFSH_ALPHA5_ZETA02 * chisq +\
             BBHNEWPHENOMCOEFFSH_ALPHA5_ZETA11 * etachi +\
             BBHNEWPHENOMCOEFFSH_ALPHA5_ZETA10 * eta + \
             BBHNEWPHENOMCOEFFSH_ALPHA5_ZETA20 * etasq;

    alpha6 = BBHNEWPHENOMCOEFFSH_ALPHA6_ZETA01 * chi + \
             BBHNEWPHENOMCOEFFSH_ALPHA6_ZETA02 * chisq +\
             BBHNEWPHENOMCOEFFSH_ALPHA6_ZETA11 * etachi +\
             BBHNEWPHENOMCOEFFSH_ALPHA6_ZETA10 * eta + \
             BBHNEWPHENOMCOEFFSH_ALPHA6_ZETA20 * etasq;

    gamma1 = BBHNEWPHENOMCOEFFSH_GAMMA1_ZETA01 * chi + \
             BBHNEWPHENOMCOEFFSH_GAMMA1_ZETA02 * chisq +\
             BBHNEWPHENOMCOEFFSH_GAMMA1_ZETA11 * etachi +\
             BBHNEWPHENOMCOEFFSH_GAMMA1_ZETA10 * eta + \
             BBHNEWPHENOMCOEFFSH_GAMMA1_ZETA20 * etasq;

    delta1 = BBHNEWPHENOMCOEFFSH_DELTA1_ZETA01 * chi + \
             BBHNEWPHENOMCOEFFSH_DELTA1_ZETA02 * chisq +\
             BBHNEWPHENOMCOEFFSH_DELTA1_ZETA11 * etachi +\
             BBHNEWPHENOMCOEFFSH_DELTA1_ZETA10 * eta + \
             BBHNEWPHENOMCOEFFSH_DELTA1_ZETA20 * etasq;

    delta2 = BBHNEWPHENOMCOEFFSH_DELTA2_ZETA01 * chi + \
             BBHNEWPHENOMCOEFFSH_DELTA2_ZETA02 * chisq +\
             BBHNEWPHENOMCOEFFSH_DELTA2_ZETA11 * etachi +\
             BBHNEWPHENOMCOEFFSH_DELTA2_ZETA10 * eta + \
             BBHNEWPHENOMCOEFFSH_DELTA2_ZETA20 * etasq;

    return(alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, gamma1, delta1, delta2)


#******************************************************************************
#* Window functions Eq 5.8                                                    *
#******************************************************************************

def TanhWindowPlus ( freq, f0, sigma):

  fact = 4.0 * (freq-f0) / sigma;
  out = (1.0 + math.tanh(fact)) / 2.0;
  return(out);


def TanhWindowMinus ( freq, f0, sigma):

  fact = 4.0 * (freq-f0) / sigma;
  out = (1.0 - math.tanh(fact)) / 2.0;
  return(out);



def Amp(m1,m2,a1,a2,fr):

    Mz=m1+m2

    if(a1>0.85):
        a1=0.85
    if(a2>0.85):
        a2=0.85

    chi0 = (a1*m1+a2*m2)/(m1+m2)
    eta0 = min(m1*m2/(m1+m2)**2,0.25)

    etaMin=0.25/1.25**2 #simulations of 0710.3345v3 only have comparable mass ratios

    if eta0>etaMin :
        Amp = AmpPhenomC(fr, eta0, chi0, Mz)
    else:
        # When eta < etaMin , compute h for eta=etaMin and rescale it
        #print "Warning: out of calibrated range"
        RescaleFact = np.sqrt(eta0/etaMin)
        Amp = RescaleFact * AmpPhenomC(fr, etaMin, chi0, Mz)

    return Amp*mtsun*Rs



def Aeff(f,m1z,m2z,dL,chi1,chi2): #masses in Msun, dL in Mpc
    coeff=np.sqrt(15/np.pi)/8.
    dLm=dL*Mpc
    return Amp(m1z, m2z, chi1, chi2, f)* coeff*(m1z + m2z)**2/dLm

def fMax(m1z,m2z,dL,chi1,chi2):
    MM=m1z + m2z
    eta=min(m1z*m2z/MM**2,0.25)
    a3 = 8.4845e-1
    b3 = 1.2848e-1
    c3 = 2.7299e-1

    frM=(a3*eta**2 + b3*eta + c3)/(np.pi*G*MM*Msun/c**3)
    for i in range(1000000):
        if Aeff(frM,m1z,m2z,dL,chi1,chi2)>0:
            frM=frM*1.1
        else:
            return frM





# Coefficients
a0 = 2.9740e-1; b0 = 4.4810e-2; c0 = 9.5560e-2
a1 = 5.9411e-1; b1 = 8.9794e-2; c1 = 1.9111e-1
a2 = 5.0801e-1; b2 = 7.7515e-2; c2 = 2.2369e-2
a3 = 8.4845e-1; b3 = 1.2848e-1; c3 = 2.7299e-1

def fmerger(eta, M):
    return (a0 * eta**2 + b0 * eta + c0) / (np.pi * G * M / c**3)

def fring(eta, M):
    return (a1 * eta**2 + b1 * eta + c1) / (np.pi * G * M / c**3)

def sigma(eta, M):
    return (a2 * eta**2 + b2 * eta + c2) / (np.pi * G * M / c**3)

def fcut(eta, M):
    return (a3 * eta**2 + b3 * eta + c3) / (np.pi * G * M / c**3)

def L(f, eta, M):
    s = sigma(eta, M)
    fr = fring(eta, M)
    return 1 / (2 * np.pi) * s / ((f - fr)**2 + s**2 / 4)

def AeffphenomA(f, m1z, m2z, dL): #masses in Msun, dL in Mpc
    d=dL*Mpc
    M=m1z+m2z
    eta=m1z*m2z/M**2
    M=M*Msun
    fm = fmerger(eta, M)
    fr = fring(eta, M)
    s = sigma(eta, M)
    fc = fcut(eta, M)
    w = np.pi * s / 2 * (fr / fm)**(-2/3)
    i = np.pi / 3
    CC = c * (G * M / c**3)**(5/6) * fm**(-7/6) / (d * np.pi**(2/3)) * np.sqrt(5 * eta / 24) * np.sin(i)

    if f < fm:
        return CC * (f / fm)**(-7/6)
    elif fm <= f < fr:
        return CC * (f / fm)**(-2/3)
    elif fr <= f < fc:
        return CC * w * L(f, eta, M)
    else:
        return 0  # Assuming Aeff is 0 for f >= fcut





