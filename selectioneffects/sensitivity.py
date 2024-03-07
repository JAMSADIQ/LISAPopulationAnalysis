import numpy as np
def Sgal(f,tc): #tc in yr
    Amp = 0.5*1.8e-44

    # Fix the parameters of the confusion noise fit
    if (tc < 0.75):
        est = 1
    elif (0.75 < tc and tc < 1.5):
        est = 2
    elif (1.5 < tc and tc < 3.):
        est = 3
    else:
        est = 4

    if (est==1):
        alpha  = 0.133
        beta   = 243.
        kappa  = 482.
        gamma  = 917.
        fk = 2.58e-3
    elif (est==2):
        alpha  = 0.171
        beta   = 292.
        kappa  = 1020.
        gamma  = 1680.
        fk = 2.15e-3
    elif (est==3):
        alpha  = 0.165
        beta   = 299.
        kappa  = 611.
        gamma  = 1340.
        fk = 1.73e-3
    else:
        alpha  = 0.138
        beta   = -221.
        kappa  = 521.
        gamma  = 1680.
        fk = 1.13e-3

    return Amp*f**(-7/3)*np.exp(-f**alpha + beta*f*np.sin(kappa*f))*(1 + np.tanh(gamma*(fk - f)))


def Soms(f):
    pm = 1.e-12
    return (15*pm)**2*(1 + (2.e-3/f)**4)


def Sacc(f):
    return (3.e-15)**2*(1 + (0.4e-3/f)**2)*(1 + (f/(8.e-3))**4)

L0 = 2.5e9
fstar = 19.09e-3

def SnSA(f):
    return 10/3*(2.*(1. + np.cos(f/fstar)**2)*Sacc(f)/(2*np.pi*f)**4 + Soms(f))/L0**2*(1 + 6/10*(f/fstar)**2)

def Sn(f,tc):
    return 3/10*(SnSA(f) + Sgal(f,tc))

