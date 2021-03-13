# QPy
 QPy is a python module that aims at making many quantum mechanical computations/simulations operable in no more than a line of code. For example, the tise.solve() feature solves the Time-Independent Schrödinger Equation, plots the Wavefunction and prints the eigenvalues (something that would usually take a few hundred lines of code to do). This numerical package is useful for anyone taking QM1 as the syntax barely requires a learning curve and makes rigorous problems solved in seconds!


# State of the project
The first version includes 4 main features: QSolve, which solves a few algebraic QM-related eqns, QPlot, which plots a few algebraic eqns, tise, which solves the 1D Time Independant Schrodinger Equation and psiTools, which allows one to make a few wavefunction-related calculations such as normalization, finding probability, expectation values, uncertainty, etc.


This is how you import each feature:

-from QPy import Solver as QSolve

-from QPy import Plotter as QPlot

-from QPy import tise 

-from QPy import psiTools as psi



Demos are available on test.py

# Modules used

-Numpy

-Scipy

-Matplotlib

