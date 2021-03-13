from os import system
import scipy 
from scipy.integrate import quad
from scipy import integrate
import numpy as np
from scipy.misc import derivative as yo


PLANCK = 6.6*(10**(-34))
C = 299792458
E = 2.71828
KAPPA = 1.38064852*(10**(-23))
PI = np.pi 
inf = np.inf

def sqrt(x):
    bro = x**0.5
    return bro

def sin(x):
    sin = np.sin(x)
    return sin 

def exp(x):
    take = E**x
    return take


def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return scipy.real(func(x))
    def imag_func(x):
        return scipy.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0])

a = complex_quadrature(lambda x: complex(3,2*x)*complex(3,-2*x),0 , 3)

print(a)


def nParam(func):
    val = func.__code__.co_argcount
    return val 

b = nParam(lambda x,y: x+y)

print(b)

def normalize(psi,lBound = -np.inf, rBound = np.inf, funcType='real'):
        if funcType.lower() == 'real':
            if nParam(psi) == 1:
                a = integrate.quad(lambda x: psi(x)**2, lBound, rBound)
                return 1/np.sqrt(float(a[0]))
            if nParam(psi) == 2:
                a = integrate.dblquad(lambda x,y: psi(x,y)**2, lBound[0], rBound[0], lBound[1], rBound[1])
                return 1/np.sqrt(float(a[0]))
            if nParam(psi) == 3: 
                a = integrate.tplquad(lambda x,y,z: psi(x,y,z)**2, lBound[0], rBound[0], lBound[1], rBound[1], lBound[2], rBound[2])
                return 1/np.sqrt(float(a[0]))

def normalize2(psi,lBound = -np.inf, rBound = np.inf, funcType='real'):
        if funcType.lower() == 'real':
            if type(lBound) == list:
                if len(lBound) == len(rBound):
                    if len(lBound) == 1:
                        a = integrate.quad(lambda x: psi(x)**2,lBound[0], rBound[0])
                        return 1/np.sqrt(float(a[0]))
                    if len(lBound) == 2:
                        a = integrate.dblquad(lambda x,y: psi(x,y)**2,lBound[0], rBound[0],lBound[1], rBound[1])
                        return 1/np.sqrt(float(a[0]))
                    if len(lBound) == 3:
                        a = integrate.tplquad(lambda x,y,z: psi(x,y,z)**2,lBound[0], rBound[0],lBound[1], rBound[1], lBound[2],rBound[2])
                        return 1/np.sqrt(float(a[0]))
            else:
                a = integrate.quad(lambda x: psi(x)**2,lBound, rBound)
                return 1/np.sqrt(float(a[0]))

b = normalize(lambda x,y,z: np.sin(PI*x)*np.sin(PI*y)*np.sin(PI*z), [0,0,0], [1,1,1])

A = 1
BETA = 1


print(b)

                        
from matplotlib import pyplot as plt 

#Wein's Law: solve A and B problem, fix y-scaling issue, range(wein) = range(planck)/2

def blackBody(temp,lim1=0,lim2="default", law="planck", title='Power Density Distribution vs Wavelength'):
            
        if law == 'planck':

            if lim2 == "default":
                if type(temp) == list:
                    n = 0
                    while n < len(temp):
                        x = [i for i in np.linspace(lim1, np.mean(temp)/2.5,500000)]
                        y = [(2*PI*PLANCK*((10**9)*C/i)**5)/(C**3*(E**((PLANCK*(10**9)*(C/i))/(KAPPA*temp[n])) - 1))*10**(-13) for i in x]
                        y2 = [A*(i**5)*(E**-BETA*i/temp[n]) for i in x]
                        plt.plot(x,y2, label = 'Weins law')
                        plt.plot(x,y, label='Temp = ' + str(temp[n]) + "K")
                        n += 1 
                else:
                    x = [i for i in np.linspace(lim1, ((np.mean(temp)/2.5)),500000)]
                    y = [(2*PI*PLANCK*((10**9)*C/i)**5)/(C**3*(E**((PLANCK*(10**9)*(C/i))/(KAPPA*temp)) - 1))*10**(-13) for i in x]
                    y2 = [A*(i**5)*(E**-BETA*i/temp) for i in x]
                    plt.plot(x,y2, label = 'Weins law')
                    plt.plot(x,y, label='Temp = ' + str(temp) + "K")
            else:
                if type(temp) == list:
                    n = 0
                    while n < len(temp):
                        x = [i for i in np.linspace(lim1, lim2,500000)]
                        y = [(2*PI*PLANCK*((10**9)*C/i)**5)/(C**3*(E**((PLANCK*(10**9)*(C/i))/(KAPPA*temp[n])) - 1))*10**(-13) for i in x]
                        y2 = [A*(i**5)*(E**-BETA*i/temp[n]) for i in x]
                        plt.plot(x,y2, label = 'Weins law')
                        plt.plot(x,y, label='Temp = ' + str(temp[n]) + "K")
                        n += 1 
                else:
                    x = [i for i in np.linspace(lim1, lim2, 500000)]
                    y = [(2*PI*PLANCK*((10**9)*C/i)**5)/(C**3*(E**((PLANCK*(10**9)*(C/i))/(KAPPA*temp)) - 1))*10**(-13) for i in x]
                    y2 = [A*(i**5)*(E**-BETA*i/temp) for i in x]
                    plt.plot(x,y2, label = 'Weins law')
                    plt.plot(x,y, label='Temp = ' + str(temp) + "K")
        
        return plt.title(title), plt.xlabel("Wavelength (nm)"), plt.ylabel("Power Density (10^13)"), plt.legend(), plt.show()



def normy(psi, x1 = -inf, x2 = inf, y1 = -inf, y2 = inf, z1 = -inf, z2 = inf):
    if nParam(psi) == 1:
        a = integrate.quad(lambda x: abs(psi(x)*np.conj(psi(x))), x1, x2)
        return 1/np.sqrt(float(a[0]))
    if nParam(psi) == 2:
        a = integrate.dblquad(lambda x,y: abs(psi(x,y)*np.conj(psi(x,y))), x1,x2,y1,y2)
        return 1/np.sqrt(float(a[0]))
    if nParam(psi) == 3: 
        a = integrate.tplquad(lambda x,y,z: abs(psi(x,y,z)*np.conj(psi(x,y,z))), x1,x2,y1,y2,z1,z2)
        return 1/np.sqrt(float(a[0]))


def normal(psi, a = -inf, b = inf):
    bro = quad(lambda x: psi(x)**2,a,b)
    return 1/sqrt(bro)



normal(lambda x: exp(-x**2))



x = [i for i in np.linspace(0,2200, 1000000)]

planck = [(2*PI*PLANCK*((10**9)*C/i)**5)/(C**3*(E**((PLANCK*(10**9)*(C/i))/(KAPPA*5500)) - 1))*10**(-13) for i in x]
rJeans = [8*PI*KAPPA*5500*(10**9)/((C/i)**4) for i in x]


plt.plot(x,planck)

plt.plot(x,rJeans)

plt.show()








