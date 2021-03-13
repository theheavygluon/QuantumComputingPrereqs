#Demo of some features. 
#This scripts demonstate a total of 10 features that. You can reduce this to 7 lines of code!

from QPy import *
from QPy import Solver as QSolve
from QPy import Plotter as QPlot 
from QPy import tise
from QPy import psiTools as psi 

#QSolve feature example: Solving f = (KE + phi)/h for KE = 6000, phi = 5000 
print(QSolve.photoelectric.frequency(6000,5000))

#QPlot feature example: Plotting the Spectral Radiance vs Wavelength(~Intensity) for temps 5000,5500,6500 using default domain
QPlot.blackBody([5000,5500,6500])

#Getting the first 3 solutions of the TISE for the harmonic oscillator (note that tise.harmOsc(1,3) would give the exact same result)
tise.solve("0.5*x**2", 3)

#Normalizing psi(x) = x*e^(-x^2)
print(psi.normalize(lambda x: x*exp(-x**2)))

#Normalizing the wavefunction solution of the infinite cubic well where l = 1, n = 1 
#(Since l = 1, the x,y,z boundaries are +/- l/2 or +/- 0.5)

print(psi.normalize(lambda x,y,z: sin(PI*x)*sin(PI*y)*sin(PI*z), -0.5, 0.5, -0.5, 0.5, -0.5, 0.5))


#Plotting the probability of finding a particle defined by psi(x) = cos(PI*x) between it's left barrier (-0.5) and x 

x = [i for i in np.linspace(-0.5,0.5, 10000)] 
y = [psi.prob(lambda x: cos(PI*x), -0.5,i, -0.5, 0.5) for i in x]

plt.plot(x,y, color='r', label='$\int_{-0.5}^{x} |\psi(x)^2|$, where $\psi = \cos(\pi x)$')

plt.title('Plotting the probability density of finding the particle between -0.5 and x \n (Because its normalized, $Prob_{(-0.5,0.5) = 1}$)')
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$Prob_{(-0.5,x)}$')
plt.show()

#Postion Operators, Expectation and Sigma Values 
print("psi(x) = x*exp(-x^2)")
print("Position operator at x = 1: " + str(psi.hat.x(lambda x: x*exp(-x**2))(1)))
print("<x> = " + str(psi.x.expVal(lambda x: x*exp(-x**2))))
print("sigma_x = " + str(psi.x.sigma(lambda x: x*exp(-x**2))))
