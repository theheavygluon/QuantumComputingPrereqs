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



class Solver():
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

'''
def eigsTry(potential, E_level):
    from back import numerovEigs as algo
    sol = algo.Numerov(potential,E_level)
    return sol
'''


#Algebraic Plotter
#Things to work on: 
#Finish blackBody(). Fix the rJean's law, do wein, do all. Write equations for rJeans and Wein
#Compton Wavelength 
                                                                                                                                   
class Plotter():
    class photoelectric():        
        
        def energy(phi,lim1,lim2, title="Dragonflycatboi"):
            freq = [f for f in np.linspace(lim1,lim2, 10)]
            energy = [PLANCK*f - phi for f in freq]
            plt.plot(freq,energy, label='KE = hf - ' + str(int(phi)))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Kinetic Energy (Jouls)')
            if title=="Dragonflycatboi":
                plt.title("Kinetic Energy vs. Freq when \u03C6 = " + str(phi)+ "\n in range " + str(lim1) + " to " + str(lim2))
            else:
                plt.title(title)
            plt.legend()
            return plt.show()

        def frequency(phi,lim1,lim2, title='Dragonflycatboi'):
            freq = [f for f in np.linspace(lim1,lim2, 10)]
            energy = [PLANCK*f - phi for f in freq]
            plt.plot(energy,freq, label='KE = hf - ' + str(int(phi)))
            plt.xlabel('Kinetic Energy (Jouls)')
            plt.ylabel('Frequency (Hz)')
            if title=="Dragonflycatboi":
                plt.title("Freq.  vs. Kinetic Energy when \u03C6 = " + str(phi)+ "\n in range " + str(lim1) + " to " + str(lim2))
            else:
                plt.title(title)
            plt.legend()
            return plt.show()

    class deBroglie():
        def momentum(lim1,lim2, title='Dragonflycatboi'):
            lamda = [i for i in np.linspace(lim1,lim2, (lim2-lim1)*5000)]
            p = [PLANCK/i for i in lamda]
            plt.plot(lamda,p, label='p = h/\u03BB')
            plt.xlabel('Wavelength (m)')
            plt.ylabel('Momentum N.s')
            if title=="Dragonflycatboi":
                plt.title("Momentum vs. Wavelength from " + str(lim1) +  " to " + str(lim2))
            else:
                plt.title(title)
            plt.legend()
            return plt.show()

        def wavelength(lim1, lim2, title='Dragonflycatboi'):
            lamda = [i for i in np.linspace(lim1,lim2, (lim2-lim1)*500)]
            p = [PLANCK/i for i in lamda]
            plt.plot(lamda,p, label='\u03BB = h/p')
            plt.ylabel('Wavelength (m)')
            plt.xlabel('Momentum N.s')
            if title=="Dragonflycatboi":
                plt.title("Wavelength vs. Momentum from " + str(lim1) +  " to " + str(lim2))
            else:
                plt.title(title)
            plt.legend()
            return plt.show()
        

    def blackBody(temp,lim1=0,lim2="default", law="planck", title='Power Density Distribution vs Wavelength'):
            
        if law == 'planck':

            if lim2 == "default":
                if type(temp) == list:
                    n = 0
                    while n < len(temp):
                        x = [i for i in np.linspace(lim1, np.mean(temp)/2.5,500000)]
                        y = [(2*PI*PLANCK*((10**9)*C/i)**5)/(C**3*(E**((PLANCK*(10**9)*(C/i))/(KAPPA*temp[n])) - 1))*10**(-13) for i in x]
                        plt.plot(x,y, label='Temp = ' + str(temp[n]) + "K")
                        n += 1 
                else:
                    x = [i for i in np.linspace(lim1, ((np.mean(temp)/2.5)),500000)]
                    y = [(2*PI*PLANCK*((10**9)*C/i)**5)/(C**3*(E**((PLANCK*(10**9)*(C/i))/(KAPPA*temp)) - 1))*10**(-13) for i in x]
                    plt.plot(x,y, label='Temp = ' + str(temp) + "K")
            else:
                if type(temp) == list:
                    n = 0
                    while n < len(temp):
                        x = [i for i in np.linspace(lim1, lim2,500000)]
                        y = [(2*PI*PLANCK*((10**9)*C/i)**5)/(C**3*(E**((PLANCK*(10**9)*(C/i))/(KAPPA*temp[n])) - 1))*10**(-13) for i in x]
                        plt.plot(x,y, label='Temp = ' + str(temp[n]) + "K")
                        n += 1 
                else:
                    x = [i for i in np.linspace(lim1, lim2, 500000)]
                    y = [(2*PI*PLANCK*((10**9)*C/i)**5)/(C**3*(E**((PLANCK*(10**9)*(C/i))/(KAPPA*temp)) - 1))*10**(-13) for i in x]
                    plt.plot(x,y, label='Temp = ' + str(temp) + "K")

        if law == 'wein':

            if lim2 == "default":
                if type(temp) == list:
                    n = 0
                    while n < len(temp):
                        x = [i for i in np.linspace(lim1, np.mean(temp)/2.5,500000)]
                        y = [2*i for i in x]
                        plt.plot(x,y, label='Temp = ' + str(temp[n]) + "K")
                        n += 1 
                else:
                    x = [i for i in np.linspace(lim1, ((np.mean(temp)/2.5)),500000)]
                    y = [2*i for i in x]
                    plt.plot(x,y, label='Temp = ' + str(temp) + "K")
            else:
                if type(temp) == list:
                    n = 0
                    while n < len(temp):
                        x = [i for i in np.linspace(lim1, lim2,500000)]
                        y = [2*i for i in x]
                        plt.plot(x,y, label='Temp = ' + str(temp[n]) + "K")
                        n += 1 
                else:
                    x = [i for i in np.linspace(lim1, lim2, 500000)]
                    y = [2*i for i in x]
                    plt.plot(x,y, label='Temp = ' + str(temp) + "K")


        if law == 'rJeans':

            if lim2 == "default":
                if type(temp) == list:
                    n = 0
                    while n < len(temp):
                        x = [i for i in np.linspace(0.0001, np.mean(temp)/2500,500000)]
                        y = [2*PI*(((10**9)*C/i)**4)*KAPPA*temp[n]/(C**3) for i in x]
                        plt.plot(x,y, label='Temp = ' + str(temp[n]) + "K")
                        n += 1 
                else:
                    x = [i for i in np.linspace(lim1, ((np.mean(temp)/2500)),500000)]
                    y = [2*PI*(((10**9)*C/i)**4)*KAPPA*temp/(C**3) for i in x]
                    plt.plot(x,y, label='Temp = ' + str(temp) + "K")
            else:
                if type(temp) == list:
                    n = 0
                    while n < len(temp):
                        x = [i for i in np.linspace(lim1, lim2,500000)]
                        y = [2*PI*(((10**9)*C/i)**4)*KAPPA*temp[n]/(C**3) for i in x]
                        plt.plot(x,y, label='Temp = ' + str(temp[n]) + "K")
                        n += 1 
                else:
                    x = [i for i in np.linspace(lim1, lim2, 500000)]
                    y = [2*PI*(((10**9)*C/i)**4)*KAPPA*temp/(C**3) for i in x]
                    plt.plot(x,y, label='Temp = ' + str(temp) + "K")

        
        
        return plt.title(title), plt.xlabel("Wavelength (nm)"), plt.ylabel("Power Density (10^13)"), plt.legend(), plt.show()

#Time Independent Schrodinger Equation

class tise:
    def solve(potential, E_level, alg='numerov'):
        if alg.lower() == "numerov":
            from back import numerovMethod as algo
            sol = algo.Numerov(potential,E_level)
        return sol

    
    def harmOsc(k,n, alg='numerov'):
        if alg.lower() == "numerov":
            from back import numerovMethod as algo
            sol = algo.Numerov("0.5*" + str(k) + "*x**2", n)
            return sol


    def eigs(potential, E_level, showWork=False):
        from back import numerovMethod as algo
        sol = algo.eigs(potential,E_level, showWork)
        return sol



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

