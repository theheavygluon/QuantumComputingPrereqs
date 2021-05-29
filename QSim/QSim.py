import numpy as np
from functools import reduce
import math 
from sympy import Matrix, init_printing
import random
from numpy.random import choice
from collections import Counter

# Matrices
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.identity(2)
H = 1/np.sqrt(2)*np.array([[1, 1], [1, -1]])
CX = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]])
CNOT = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]])
zero = np.array([[1],[0]])
one = np.array([[0],[1]])

#CX Unitary of a circuit

def view(mat):
    display(Matrix(mat))

def decimalToBinary(n): 
    return bin(n).replace("0b", "")


tensor = lambda *initial_state: reduce(lambda x, y: np.kron(x, y), initial_state)

def cx_unitary(num_qubits, control, target):
    I = np.identity(2)
    left = [I]*num_qubits
    right = [I]*num_qubits
    left[control] = np.dot(zero, zero.T)
    right[control] = np.dot(one, one.T)
    right[target] = X
    return tensor(*left) + tensor(*right)

# Eigenvectors of Pauli Matrices

plus = np.array([[1], [1]])/np.sqrt(2) # X plus basis state
minus = np.array([[1], [-1]])/np.sqrt(2) # X minus basis state

up = np.array([[1], [1j]])/np.sqrt(2) # Y plus basis state
down = np.array([[1], [-1j]])/np.sqrt(2) # Y plus basis state

# Bell States
B00 = np.array([[1], [0], [0], [1]])/np.sqrt(2) # Bell of 00
B01 = np.array([[1], [0], [0], [-1]])/np.sqrt(2) # Bell of 01
B10 = np.array([[0], [1], [1], [0]])/np.sqrt(2) # Bell of 10
B11 = np.array([[0], [-1], [1], [0]])/np.sqrt(2) # Bell of 11

# Rn Matrix Function
Rx = lambda theta: np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]])
Ry = lambda theta: np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])
Rz = lambda theta: np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]])



# return vector of size 2**num_qubits with all zeroes except first element which is 1
def get_ground_state(n):
    zero = [1,0]
    i = 0
    state = [1,0]
    while i<n-1:
        state = np.kron(state,zero)
        i+=1
    return state



# return unitary operator of size 2**n x 2**n for given gate and target qubits


def get_operator(total_qubits, gate_unitary, target_qubits):
    if type(target_qubits) == int:
        state = [I]*total_qubits
        state[target_qubits] = gate_unitary
        state = tensor(*state)
    if type(target_qubits) == list:
        if gate_unitary.all() == np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]]).all() or gate_unitary == [[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]].all():
            state = cx_unitary(total_qubits, target_qubits[0], target_qubits[1])
    return state

def view_operator(total_qubits, gate_unitary, target_qubits):
    if type(target_qubits) == int:
        state = [I]*total_qubits
        state[target_qubits] = gate_unitary
        state = tensor(*state)
    if type(target_qubits) == list:
        if gate_unitary.all() == np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]]).all() or gate_unitary == [[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]].all():
            state = cx_unitary(total_qubits, target_qubits[0], target_qubits[1])
    return view(state)


def run_program(initial_state, circuit):
    i = 0
    state = np.array(initial_state)
    while i< len(circuit):
        state = np.dot(get_operator(int(math.log2(len(initial_state))), circuit[i]['gate'],circuit[i]['target']),state)
        i+=1
    return state


    
def measure(state_vector):
    i = 0
    
    while i < len(state_vector):
        print(str(decimalToBinary(i)) + ": " + str(round(state_vector[i]**2, 2)))
        i+=1


def get_counts(state_vector, num_shots):
    size = [i for i in range(len(state_vector))]
    weights = [i**2 for i in state_vector]
    states = [str(decimalToBinary(i)) for i in size]
    draw = choice(states, num_shots, p=weights)
    result = [list(draw).count(i)/num_shots for i in states]
    return result
