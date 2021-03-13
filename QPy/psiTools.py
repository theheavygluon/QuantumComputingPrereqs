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


def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return scipy.real(func(x))
    def imag_func(x):
        return scipy.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

def nParam(func):
    val = func.__code__.co_argcount
    return val 




class psiTools():
    

    def normalize(psi, x1 = -inf, x2 = inf, y1 = -inf, y2 = inf, z1 = -inf, z2 = inf):
        if nParam(psi) == 1:
            a = integrate.quad(lambda x: abs(psi(x)*np.conj(psi(x))), x1, x2)
            return 1/np.sqrt(float(a[0]))
        if nParam(psi) == 2:
            a = integrate.dblquad(lambda x,y: abs(psi(x,y)*np.conj(psi(x,y))), x1,x2,y1,y2)
            return 1/np.sqrt(float(a[0]))
        if nParam(psi) == 3: 
            a = integrate.tplquad(lambda x,y,z: abs(psi(x,y,z)*np.conj(psi(x,y,z))), x1,x2,y1,y2,z1,z2)
            return 1/np.sqrt(float(a[0]))


    def prob(psi,lBound, rBound, lNorm=-inf, rNorm=inf):
        if nParam(psi) == 1:
            b = psiTools.normalize(lambda x: psi(x), lNorm,rNorm)
        if nParam(psi) == 2:
            b = psiTools.normalize(lambda x,y: psi(x,y), lNorm, rNorm)
        if nParam(psi) == 3:
            b = psiTools.normalize(lambda x,y,z: psi(x,y,z), lNorm, rNorm)
        if nParam(psi) == 1:
            a = quad(lambda x: (b*psi(x))**2,lBound, rBound)
        if nParam(psi) == 2: 
            a = integrate.dblquad(lambda x,y: (b*psi(x,y))**2,lBound[0], rBound[0], lBound[1], rBound[1])
        if nParam(psi) == 3:
            a = integrate.tplquad(lambda x,y,z: (b*psi(x,y,z))**2,lBound[0], rBound[0], lBound[1], rBound[1], lBound[2], rBound[2])
        return a[0]

    class hat():
        
        def x(psi, lBound = -inf, rBound = inf):
            a = psiTools.normalize(psi, lBound, rBound)
            pos = lambda x: a*x*psi(x)
            return pos
        
        def y(psi, lBound = -inf, rBound = inf):
            a = psiTools.normalize(psi, lBound, rBound)
            pos = lambda y: a*y*psi(y)
            return pos
         
        def z(psi, lBound = -inf, rBound = inf):
            a = psiTools.normalize(psi, lBound, rBound)
            pos = lambda z: a*z*psi(z)
            return pos
        
        def x2(psi, lBound = -inf, rBound = inf):
            a = psiTools.normalize(psi, lBound, rBound)
            pos = lambda x: a*(x**2)*psi(x)
            return pos
        
        def y2(psi, lBound = -inf, rBound = inf):
            a = psiTools.normalize(psi, lBound, rBound)
            pos = lambda y: a*(y**2)*psi(y)
            return pos
         
        def z2(psi, lBound = -inf, rBound = inf):
            a = psiTools.normalize(psi, lBound, rBound)
            pos = lambda z: a*(z**2)*psi(z)
            return pos
        
        def p(psi, lBound = -inf, rBound = inf):
            a = psiTools.normalize(psi, lBound, rBound)
            momentum = derivative(lambda x:-HBAR*a*psi(x)*1j)
            return momentum

        def px(psi, lBound = -inf, rBound = inf):
            a = psiTools.normalize(psi, lBound, rBound)
            momentum = derivative(lambda x:-HBAR*a*psi(x)*1j)
            return momentum
        
        def py(psi, lBound = -inf, rBound = inf):
            a = psiTools.normalize(psi, lBound, rBound)
            momentum = derivative(lambda x:-HBAR*a*psi(x)*1j)
            return momentum

        def pz(psi, lBound = -inf, rBound = inf):
            a = psiTools.normalize(psi, lBound, rBound)
            momentum = derivative(lambda x:-HBAR*a*psi(x)*1j)
            return momentum
        
        #def p2d():
        #def p3d():

    class x():
            
            def expVal(psi, lBound = -inf, rBound = inf):
                a = psiTools.normalize(psi, lBound, rBound)
                exp = quad(lambda x: (a**2)*np.conj(psi(x))*x*psi(x), lBound, rBound)
                return exp[0]
            
            def sigma(psi, lBound = -inf, rBound = inf):
                a = psiTools.normalize(psi, lBound, rBound)
                expX = quad(lambda x: (a**2)*np.conj(psi(x))*x*psi(x), lBound, rBound)
                expX2 = quad(lambda x: (a**2)*np.conj(psi(x))*(x**2)*psi(x), lBound, rBound)
                var = expX2[0] - expX[0]**2
                return sqrt(var)


    class y():
            
            def expVal(psi, lBound = -inf, rBound = inf):
                a = psiTools.normalize(psi, lBound, rBound)
                exp = quad(lambda y: (a**2)*np.conj(psi(y))*y*psi(y), lBound, rBound)
                return exp[0]
            def sigma(psi, lBound = -inf, rBound = inf):
                a = psiTools.normalize(psi, lBound, rBound)
                expX = quad(lambda y: (a**2)*np.conj(psi(y))*y*psi(y), lBound, rBound)
                expX2 = quad(lambda y: (a**2)*np.conj(psi(y))*(y**2)*psi(y), lBound, rBound)
                var = expX2[0] - expX[0]**2
                return sqrt(var)
    
    class z():
            
            def expVal(psi, lBound = -inf, rBound = inf):
                a = psiTools.normalize(psi, lBound, rBound)
                exp = quad(lambda z: (a**2)*np.conj(psi(z))*z*psi(z), lBound, rBound)
                return exp[0]
            def sigma(psi, lBound = -inf, rBound = inf):
                a = psiTools.normalize(psi, lBound, rBound)
                expX = quad(lambda z: (a**2)*np.conj(psi(z))*z*psi(z), lBound, rBound)
                expX2 = quad(lambda z: (a**2)*np.conj(psi(z))*(z**2)*psi(z), lBound, rBound)
                var = expX2[0] - expX[0]**2
                return sqrt(var)
   


    class x2():
            
            def expVal(psi, lBound = -inf, rBound = inf):
                a = psiTools.normalize(psi, lBound, rBound)
                exp = quad(lambda x: (a**2)*np.conj(psi(x))*(x**2)*psi(x), lBound, rBound)
                return exp[0]
            def sigma(psi, lBound = -inf, rBound = inf):
                a = psiTools.normalize(psi, lBound, rBound)
                expX = quad(lambda x: (a**2)*np.conj(psi(x))*(x**2)*psi(x), lBound, rBound)
                expX2 = quad(lambda x: (a**2)*np.conj(psi(x))*(x**4)*psi(x), lBound, rBound)
                var = expX2[0] - expX[0]**2
                return sqrt(var)

    class y2():
            
            def expVal(psi, lBound = -inf, rBound = inf):
                a = psiTools.normalize(psi, lBound, rBound)
                exp = quad(lambda y: (a**2)*np.conj(psi(y))*(y**2)*psi(y), lBound, rBound)
                return exp[0]
            def sigma(psi, lBound = -inf, rBound = inf):
                a = psiTools.normalize(psi, lBound, rBound)
                expX = quad(lambda y: (a**2)*np.conj(psi(y))*(y**2)*psi(y), lBound, rBound)
                expX2 = quad(lambda y: (a**2)*np.conj(psi(y))*(y**4)*psi(y), lBound, rBound)
                var = expX2[0] - expX[0]**2
                return sqrt(var)
    
    class z2():
            
            def expVal(psi, lBound = -inf, rBound = inf):
                a = psiTools.normalize(psi, lBound, rBound)
                exp = quad(lambda z: (a**2)*np.conj(psi(z))*(z**2)*psi(z), lBound, rBound)
                return exp[0]
            def sigma(psi, lBound = -inf, rBound = inf):
                a = psiTools.normalize(psi, lBound, rBound)
                expX = quad(lambda z: (a**2)*np.conj(psi(z))*(z**2)*psi(z), lBound, rBound)
                expX2 = quad(lambda z: (a**2)*np.conj(psi(z))*(z**4)*psi(z), lBound, rBound)
                var = expX2[0] - expX[0]**2
                return sqrt(var)


    class p():

        def expVal(psi, lBound = -inf, rBound = inf):
            a = psiTools.normalize(psi, lBound, rBound)
            exp = quad(lambda x: abs((a**2)*np.conj(psi(x))*derivative(lambda x:-HBAR*psi(x)*1j)), lBound, rBound)
            return exp[0]

        def sigma(psi, lBound = -inf, rBound = inf):
            a = psiTools.normalize(psi, lBound, rBound)
            expX = quad(lambda x: (a**2)*np.conj(psi(x))**derivative(lambda x:-HBAR*psi(x)*1j), lBound, rBound)
            expX2 = quad(lambda x: (a**2)*np.conj(psi(x))*(x**4)**derivative(lambda x:-HBAR*(psi(x)**2)*1j), lBound, rBound)
            var = expX2[0] - expX[0]**2
            return sqrt(var)

    class px():
        
        def expVal(psi, lBound = -inf, rBound = inf):
            a = psiTools.normalize(psi, lBound, rBound)
            exp = quad(lambda x: abs((a**2)*np.conj(psi(x))*derivative(lambda x:-HBAR*psi(x)*1j)), lBound, rBound)
            return exp[0]
        
        def sigma(psi, lBound = -inf, rBound = inf):
            a = psiTools.normalize(psi, lBound, rBound)
            expX = quad(lambda x: (a**2)*np.conj(psi(x))**derivative(lambda x:-HBAR*psi(x)*1j), lBound, rBound)
            expX2 = quad(lambda x: (a**2)*np.conj(psi(x))*(x**4)**derivative(lambda x:-HBAR*(psi(x)**2)*1j), lBound, rBound)
            var = expX2[0] - expX[0]**2
            return sqrt(var)

    class py():
        
        def expVal(psi, lBound = -inf, rBound = inf):
            a = psiTools.normalize(psi, lBound, rBound)
            exp = quad(lambda x: abs((a**2)*np.conj(psi(x))*derivative(lambda x:-HBAR*psi(x)*1j)), lBound, rBound)
            return exp[0]
            
            def sigma(psi, lBound = -inf, rBound = inf):
                a = psiTools.normalize(psi, lBound, rBound)
                expX = quad(lambda x: (a**2)*np.conj(psi(x))**derivative(lambda x:-HBAR*psi(x)*1j), lBound, rBound)
                expX2 = quad(lambda x: (a**2)*np.conj(psi(x))*(x**4)**derivative(lambda x:-HBAR*(psi(x)**2)*1j), lBound, rBound)
                var = expX2[0] - expX[0]**2
                return sqrt(var)

    class p():
        
        def expVal(psi, lBound = -inf, rBound = inf):
            a = psiTools.normalize(psi, lBound, rBound)
            exp = quad(lambda x: abs((a**2)*np.conj(psi(x))*derivative(lambda x:-HBAR*psi(x)*1j)), lBound, rBound)
            return exp[0]
            
            def sigma(psi, lBound = -inf, rBound = inf):
                a = psiTools.normalize(psi, lBound, rBound)
                expX = quad(lambda x: (a**2)*np.conj(psi(x))**derivative(lambda x:-HBAR*psi(x)*1j), lBound, rBound)
                expX2 = quad(lambda x: (a**2)*np.conj(psi(x))*(x**4)**derivative(lambda x:-HBAR*(psi(x)**2)*1j), lBound, rBound)
                var = expX2[0] - expX[0]**2
                return sqrt(var)

