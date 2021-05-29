# QuantAlgs
 QuantAlgs is an module that aims at allowing basic quantum algorithms to be solved using one line of code only. The first version integrates the following algorithms: Bernstein-Vazarani (berVaz), Deutsch–Jozsa (deuJoz), Grover Search (groveSearch) and Shor's Factoring Algorithm (shor). This is pretty much an extension to the berVaz module I made around a week prior to this one.
 

# Example script:
 
 -------------------------------------------------------------------------------------------------------------------------------------------------------------------

from QuantAlgs import *

deuJoz.sim(3, "b") #Simulates a three bit number and a balanced function using the qasm_simulator

deuJoz.run(3,"b", device='ibmq_16_melbourne', plotTool='mpl', shots=1000) #Runs the algorithm above but on the ibmq_16_mealbourne quantum computer with 1000 shots

berVaz.sim('10011') #Runs the Bernstein-Vazarani algorithm on input '1011' using the qasm_simulator

berVaz.run('1001', device='ibmq_16_mealbourne') #Runs the Bernstein Vazarani algorithm for the number '1001' on the ibmq_16_mealbourne quantum computer. 

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

QuantAlgs was created for the paper I'm writing, "Introduction to Quantum Algorithms using QuantAlgs"

Note that this is the first release, where only 2/4 algorithms have been completely written. I'll write the proper documentation with all the features when all four algs are released (hopefully in a few days). 

# Things left to do

-Grover Search

-Shor




 
