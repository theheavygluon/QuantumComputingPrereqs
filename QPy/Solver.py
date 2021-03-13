from matplotlib import pyplot as plt 
import numpy as np 
import scipy 
from scipy import integrate
from scipy.integrate import quad

PI = np.pi 
PLANCK = 6.6*(10**(-34))
H = PLANCK
HBAR = H/(2*PI)
C = 299792458
E = 2.71828
KAPPA = 1.38064852*(10**(-23))

def der(f):
    h = 1/1000000
    slope = lambda x: (f(x+ h) - f(x))/h
    return slope
def derivative(psi):
    h = 1e-11
    slope = lambda x: (psi(x+h)-psi(x))/h
    return slope

inf = np.inf

def sqrt(x):
    return x**0.5
def sin(x):
    return np.sin(x)
def cos(x):
    return np.cos(x)
def tan(x):
    return np.tan(x)
def sec(x):
    return np.sec(x)
def csc(x):
    return 1/sin(x)
def cot(x):
    return 1/tan(x)
def exp(x):
    return E**(x)
def e(x):
    return E**(x)



#Algebraic Solver
#Things to work on
#blackBody (verify eqn.)
#Compton 




class photoelectric():
    def energy(f,phi):
        energy = PLANCK*f - phi
        return energy

    def frequency(e,phi):
        freq = (e + phi)/PLANCK
        return freq

    def bindingE(e,f):
        workfunc = PLANCK*f - e
        return workfunc

    def threshold(phi):
        thresh = phi/PLANCK
        return thresh

class deBroglie():
    def wavelength(p):
        lamda = PLANCK/p
        return lamda
    def momentum(lamda):
        p = PLANCK/lamda
        return p

class blackBody():
    def intensity(freq,temp):
        u = (2*PI*PLANCK*(freq**5))/((C**3)*(E**((PLANCK*freq)/(KAPPA*temp))-1))
        return u
