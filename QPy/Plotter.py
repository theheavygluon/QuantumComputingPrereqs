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
