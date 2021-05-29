from qiskit import *
from qiskit.tools.visualization import plot_histogram
from matplotlib import pyplot as plt
import numpy as np
import random as random

PI = np.pi
E = np.e

class berVaz():

#The sim function simulates the circuit on the qasm_simulator and gives 2 outputs: the circuit
# and the result of the state measurement 

    def sim(n, plotTool='text'):
        N = [int(x) for x in str(n)]
        circuit = QuantumCircuit(len(n)+1,len(n))
        circuit.h([i for i in range(len(n))])
        circuit.x(len(n))
        circuit.h(len(n))

        circuit.barrier()


        i = 0
        while i < len(n):
            if N[i] == 1:
                circuit.cx(len(n) - (i+1),len(n))
            i+=1
            
        circuit.barrier()
        circuit.h([i for i in range(len(n))])
        circuit.barrier()
        circuit.measure([i for i in range(len(n))],[i for i in range(len(n))])
        backend = Aer.get_backend('qasm_simulator')
        result = execute(circuit,backend, shots = 1).result()
        output = result.get_counts()
        return circuit.draw(plotTool), output
    

#The run function simulates the circuit on an actual IBMQ device and gives 2 outputs: the circuit and the result of the state measurement.  
#It uses the ibmq_16_melbourne system by default but you can change it by defining a device= parameter. You can also change the number of 
#shots with the shots= parameter.

    def run(n,plotTool='text',device='ibmq_16_melbourne', shots=1024):
        N = [int(x) for x in str(n)]
        circuit = QuantumCircuit(len(n)+1,len(n))
        circuit.h([i for i in range(len(n))])
        circuit.x(len(n))
        circuit.h(len(n))

        circuit.barrier()


        i = 0
        while i < len(n):
            if N[i] == 1:
                circuit.cx(len(n) - (i+1),len(n))
            i+=1
            
        circuit.barrier()
        circuit.h([i for i in range(len(n))])
        circuit.barrier()
        circuit.measure([i for i in range(len(n))],[i for i in range(len(n))])
        IBMQ.load_account()
        provider = IBMQ.get_provider('ibm-q')
        qcomp = provider.get_backend(device)
        job = execute(circuit, backend=qcomp, shots=int(shots))
        from qiskit.tools.monitor import job_monitor
        job_monitor(job)
        result = job.result()
        return circuit.draw(plotTool), result, circuit


class deuJoz():
    
    def sim(n, funcType, plotTool='text'):
        
        def gateMaker(n, funcType='text'):
            N = int(n)
            gate = QuantumCircuit(N+1)
            
            if funcType.lower() == "b":
                xNum = format(np.random.randint(1,2**N), '0'+str(n)+'b')
                for i in range(len(xNum)):
                    if xNum[i] == '1':
                        gate.x(i)
                for i in range(N):
                    gate.cx(i, N)
                for i in range(len(xNum)):
                    if xNum[i] == '1':
                        gate.x(i)

            if funcType.lower() == "c":
                
                output = np.random.randint(2)
                if output == 1:
                    gate.x(N)
            
            Gate = gate.to_gate()
            Gate.name = "Black Box" 
            return gate

        N = int(n)
        blackBox = gateMaker(n, funcType)
        circuit = QuantumCircuit(n+1, n)
        circuit.x(N)
        circuit.h(N)
        for i in range(N):
            circuit.h(i)
        circuit.append(blackBox, range(n+1))
        for i in range(N):
            circuit.h(i)
        
        for i in range(N):
            circuit.measure(i, i)

        backend = Aer.get_backend('qasm_simulator')
        result = execute(circuit,backend, shots = 1).result()
        output = result.get_counts()
        return circuit.draw(plotTool), output


    def run(n,funcType, plotTool='text',device='ibmq_16_melbourne', shots=1024):
            
            def gateMaker(n, funcType):
                N = int(n)
                gate = QuantumCircuit(N+1)
                
                if funcType.lower() == "b":
                    xNum = format(np.random.randint(1,2**N), '0'+str(n)+'b')
                    for i in range(len(xNum)):
                        if xNum[i] == '1':
                            gate.x(i)
                    for i in range(N):
                        gate.cx(i, N)
                    for i in range(len(xNum)):
                        if xNum[i] == '1':
                            gate.x(i)

                if funcType.lower() == "c":
                    
                    output = np.random.randint(2)
                    if output == 1:
                        gate.x(N)
                
                Gate = gate.to_gate()
                Gate.name = "Black Box" 
                return gate

            N = int(n)
            blackBox = gateMaker(n, funcType)
            circuit = QuantumCircuit(n+1, n)
            circuit.x(N)
            circuit.h(N)
            for i in range(N):
                circuit.h(i)
            circuit.append(blackBox, range(n+1))
            for i in range(N):
                circuit.h(i)
            
            for i in range(N):
                circuit.measure(i, i)

            IBMQ.load_account()
            provider = IBMQ.get_provider('ibm-q')
            qcomp = provider.get_backend(device)
            job = execute(circuit, backend=qcomp, shots=int(shots))
            from qiskit.tools.monitor import job_monitor
            job_monitor(job)
            result = job.result()
            return circuit.draw(plotTool), result, circuit

class grovSearch():

    def groverCircuit(target):

        target_list = [int(x) for x in str(target)] #Converts the target into a list (e.g '1001' => [1,0,0,1])
        n = len(target_list) #Length of target list (i.e nbr of qubits)
        counter = [i for i in range(n)] #List containing integers from 0 to num_qubits - 1

        #Defining a CnP gate. Note that CnP(PI) = CNZ
        def mcp(self, lam, control_qubits, target_qubit):
            from qiskit.circuit.library import MCPhaseGate
            num_ctrl_qubits = len(control_qubits)
            return self.append(MCPhaseGate(lam, num_ctrl_qubits), control_qubits[:] + [target_qubit],
                           [])
        
        #Sub-circuit 1: Hadamard on all qubits
        def hadamards(target):
            hadCirc = QuantumCircuit(n,n)
            hadCirc.h(counter)
            hadCirc.barrier()
            return hadCirc

        #Sub-circuit 2: Oracle 
        def oracle(target):
            filtered = [counter[i] for i in range(n) if target_list[i]==0] #Filtering the counter list to only the indices where target==0
            oracleCirc = QuantumCircuit(n,n)
            if filtered != []:
                oracleCirc.x(filtered) #In other words, if target only has 1s, do nothing 
            mcp(oracleCirc, np.pi, [i for i in range(n-1)],n-1)
            if filtered != []:
                oracleCirc.x(filtered) #Applying X gates to the qubits which represent 0
            oracleCirc.barrier()
            return oracleCirc

        #Sub-circuit 3: Amplifier
        def amplification(target):
            ampCirc = QuantumCircuit(n,n)
            k = round(PI*n/4 - 0.5) #Ideal number of amp-iterations 
            for i in range(k):
                ampCirc.h(counter)
                ampCirc.x(counter)
                mcp(ampCirc, np.pi, [i for i in range(n-1)],n-1)
                ampCirc.x(counter)
                ampCirc.h(counter)
                ampCirc.barrier()
            
            
            return ampCirc

        circuit = hadamards(target) + oracle(target) + amplification(target)
        circuit.measure(counter, counter)
        return circuit
        
    def sim(target,shots=1024, plotTool='mpl'):
        circuit = grovSearch.groverCircuit(target) #Creating a grover circuit for the input target
        backend = Aer.get_backend('qasm_simulator')
        result = execute(circuit,backend, shots = shots).result()
        output = result.get_counts()
        return circuit.draw(plotTool), output

    def run(target,plotTool='mpl',device='ibmq_16_melbourne', shots=1024):
        circuit = grovSearch.groverCircuit(target) #Creating a grover circuit for the input target
        IBMQ.load_account()
        provider = IBMQ.get_provider('ibm-q')
        qcomp = provider.get_backend(device)
        job = execute(circuit, backend=qcomp, shots=int(shots))
        from qiskit.tools.monitor import job_monitor
        job_monitor(job)
        result = job.result()
        return circuit.draw(plotTool), result, circuit

        
        



        
    




            




#UPDATE: WILL RESUME WORKING ON THE PROJECT NOW. CODE SHOULD HOPEFULLY BE DONE SOON!

#class shor():

