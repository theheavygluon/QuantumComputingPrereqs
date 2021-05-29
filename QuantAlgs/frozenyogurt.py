# Import

import numpy as np
import sympy
from sympy import *
from sympy.solvers.solveset import linsolve

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from sympy import Matrix, init_printing

import qiskit
from qiskit import *
from qiskit import QuantumCircuit as q
from qiskit.circuit import Parameter
from qiskit.aqua.circuits import *

# Representing Data
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator, UnitarySimulator
from qiskit.tools.visualization import plot_histogram, plot_state_city, plot_bloch_multivector

# Monitor Job on Real Machine
from qiskit.tools.monitor import job_monitor

from functools import reduce # perform sucessive tensor product

# Calculating cost
from sklearn.metrics import mean_squared_error

# Generating random unitary matrix
from scipy.stats import unitary_group

# Measure run time
import time

# Almost Equal
from numpy.testing import assert_almost_equal as aae

### Linear Algebra Tools

# Matrices
I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = 1/np.sqrt(2)*np.array([[1, 1], [1, -1]])
P = lambda theta: np.array([[1, 0], [0, np.exp(1j*theta)]])

# sqrt(X)
SX = 1/2 * np.array([[1+1j, 1-1j], [1-1j, 1+1j]])

# sqrt(Z)
S = np.array([[1, 0], [0, 1j]])

# sqrt(H)
SH = (1j/4-1/4)*np.array([[np.sqrt(2) + 2j, np.sqrt(2)], [np.sqrt(2), -np.sqrt(2)+2j]])

# 4th root of Z
T = np.array([[1, 0], [0, 1/np.sqrt(2) + 1/np.sqrt(2)*1j]])

# X power
Xp = lambda t: 1/2 * np.array([[1, 1], [1, 1]]) + np.exp(1j*np.pi*t)/(2) * np.array([[1, -1], [-1, 1]])

# H power
Hp = lambda t: np.exp(-1j*np.pi*t/2) * np.array([[np.cos(np.pi*t/2) + 1j/np.sqrt(2)* np.sin(np.pi*t/2), 1j/np.sqrt(2) * np.sin(np.pi*t/2)],
                                                   [1j/np.sqrt(2) * np.sin(np.pi*t/2), np.cos(np.pi*t/2)-1j/np.sqrt(2)* np.sin(np.pi*t/2)]])

CX = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

# Rn Matrix Function
Rx = lambda theta: np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]])
Ry = lambda theta: np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])
Rz = lambda theta: np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]])

# U3 Matrix
U3 = lambda theta, phi, lam: np.array([[np.cos(theta/2), -np.exp(1j*lam)*np.sin(theta/2)],
                                       [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*lam + 1j*phi)*np.cos(theta/2)]])

# Eigenvectors of Pauli Matrices
zero = np.array([[1], [0]]) # Z plus basis state
one = np.array([[0], [1]]) # Z plus basis state

plus = np.array([[1], [1]])/np.sqrt(2) # X plus basis state
minus = np.array([[1], [-1]])/np.sqrt(2) # X minus basis state

up = np.array([[1], [1j]])/np.sqrt(2) # Y plus basis state
down = np.array([[1], [-1j]])/np.sqrt(2) # Y plus basis state

# Bell States
B00 = np.array([[1], [0], [0], [1]])/np.sqrt(2) # Bell of 00
B01 = np.array([[1], [0], [0], [-1]])/np.sqrt(2) # Bell of 01
B10 = np.array([[0], [1], [1], [0]])/np.sqrt(2) # Bell of 10
B11 = np.array([[0], [-1], [1], [0]])/np.sqrt(2) # Bell of 11

# ndarray to list
to_list = lambda array: list(np.squeeze(array))

# Tensor Product of 2+ matrices/ vectors
tensor = lambda *initial_state: reduce(lambda x, y: np.kron(x, y), initial_state)

# Matrix Multiplicaton of 2+ matrices / vectors
mat_mul = lambda *initial_state: reduce(lambda x, y: np.dot(x, y), initial_state)

#Calculate Hermitian Conjugate

def dagger(mat):
    # Calculate Hermitian conjugate
    mat_dagger = np.conj(mat.T)

    '''# Assert Hermitian identity
    aae(np.dot(mat_dagger, mat), np.identity(mat.shape[0]))'''

    return mat_dagger


def cu_matrix(no_qubits, control, target, U, little_edian=True):
    """
    Manually build the unitary matrix for non-adjacent CX gates

    Parameters:
    -----------
    no_qubits: int
        Number of qubits in the circuit
    control: int
        Index of the control qubit (1st qubit is index 0)
    target: int
        Index of the target qubit (1st qubit is index 0)
    U: ndarray
        Target unitary matrix
    edian: bool (True: qiskit convention)
        Qubits order convention

    Returns:
    --------
    cx_out:
        Unitary matrix for CU gate
    """

    left = [I] * no_qubits
    right = [I] * no_qubits

    left[control] = np.dot(zero, zero.T)
    right[control] = np.dot(one, one.T)

    right[target] = U

    if little_edian:
        cx_out = tensor(*reversed(left)) + tensor(*reversed(right))
    else:
        cx_out = tensor(*left) + tensor(*right)

    # This returns a unitary in qiskit 'little eddian', to switch back, simply switch the target for control

    return cx_out


def angles_from_statevectors(output_statevector):
    """
    Calculate correct x, y rotation angles from an arbitrary output statevector

    Paramters:
    ----------
    output_statevector: ndarray
        Desired output state

    Returns:
    --------
    phi: float
        Angle to rotate about the y-axis [0, 2pi)
    theta: float
        Angle to rotate about the x-axis [0, 2pi)

    """

    # Extract the components
    x, z = output_statevector.real
    y, w = output_statevector.imag

    # Calculate the correct angles
    phi = 2 * np.arctan2(z, x)[0]
    theta = 2 * np.arctan2(y, z)[0]

    print(f'phi: {phi}')
    print(f'theta: {theta}')

    return phi, theta

def view(mat, rounding = 10):
    display(Matrix(np.round(mat, rounding)))


def control_unitary(circ, unitary, controls, target):
    """
    Composed a multi-controlled single unitary target gate

    Parameters:
    -----------
    circ: QuantumCircuit
        Qiskit circuit of appropriate size, no less qubit than the size of the controlled gate
    unitary: ndarray of (2, 2)
        Unitary operator for the target qubit
    controls: list
        Indices of controlled qubit on the original circuit
    target: int
        Index of target bit


    Returns:
    --------
    new_circ: QuantumCircuit
        Composed circuit with unitary target
    """

    # Get info about circuit parameters

    no_controls = len(controls)
    unitary_size = np.log2(len(unitary))

    # Build unitary circuit

    qc = QuantumCircuit(unitary_size)
    qc.unitary(unitary, range(int(unitary_size)))
    qc = qc.control(no_controls)

    # Composed the control part in the circuit

    new_circ = circ.compose(qc, (*controls, target))

    return new_circ


def control_phase(circ, angle, control_bit, target_bit, recip=True, pi_on=True):
    """
    Add a controlled-phase gate

    Parameters:
    -----------
    circ: QuantumCircuit
        Inputted circuit

    angle: float
        Phase Angle

    control_bit: int
        Index of control bit

    target_bit: int
        Index of target bit

    recip: bool (True)
        Take the reciprocal of the angle

    pi_on: bool (True)
        Multiply pi to the angle

    Returns:
    --------
    circ: QuantumCircuit
        Circuit with built-in CP

    """

    if recip:
        angle = 1 / angle
    if pi_on:
        angle *= np.pi

    circ.cp(angle, control_bit, target_bit)

    return circ

def milk(circ):
    return circ.draw('mpl')


def dtp(circ, print_details=True, nice=True, return_values=False):
    """
    Draw and/or return information about the transpiled circuit

    Parameters:
    -----------
    circ: QuantumCircuit
        QuantumCircuit to br transpiled
    print_details: bool (True)
        Print the number of u3 and cx gates used
    nice: bool (True)
        Show the transpiled circuit
    return_values: bool (True)
        Return the number of u3 and cx gates used

    Returns:
    --------
    no_cx: int
        Number of cx gates used
    no_u3: int
        Number of u3 gates used

    """

    # Transpile Circuit
    circ = transpile(circ, basis_gates=['u3', 'cx'], optimization_level=3)

    # Count operations
    gates = circ.count_ops()

    # Compute cost
    try:
        no_u3 = gates['u3']
    except:
        no_u3 = 0

    try:
        no_cx = gates['cx']
    except:
        no_cx = 0

    cost = no_u3 + 10 * no_cx

    if print_details:
        # Print Circuit Details
        print(f'cx: {no_cx}')

        print(f'u3: {no_u3}')
        print(f'Total cost: {cost}')

    if nice:
        return circ.draw('mpl')

    if return_values:
        return no_cx, no_u3


def get(circ, types='unitary', nice=True):
    """
    This function return the statevector or the unitary of the inputted circuit

    Parameters:
    -----------
    circ: QuantumCircuit
        Inputted circuit without measurement gate
    types: str ('unitary')
        Get 'unitary' or 'statevector' option
    nice: bool
        Display the result nicely option or just return unitary/statevector as ndarray

    Returns:
    --------
    out: ndarray
        Outputted unitary of statevector

    """

    if types == 'statevector':
        backend = BasicAer.get_backend('statevector_simulator')
        out = execute(circ, backend).result().get_statevector()
    else:
        backend = BasicAer.get_backend('unitary_simulator')
        out = execute(circ, backend).result().get_unitary()

    if nice:
        display(Matrix(np.round(out, 10)))
    else:
        return out


def sim(circ, visual='hist'):
    """
    Displaying output of quantum circuit

    Parameters:
    -----------
    circ: QuantumCircuit
        QuantumCircuit with or without measurement gates
    visual: str ('hist')
        'hist' (counts on histogram) or 'bloch' (statevectors on Bloch sphere) or None (get counts only)

    Returns:
    --------
    counts: dict
        Counts of each CBS state
    """

    # Simulate circuit and display counts on a histogram
    if visual == 'hist':
        simulator = Aer.get_backend('qasm_simulator')
        results = execute(circ, simulator).result()
        counts = results.get_counts(circ)
        display(plot_histogram(counts))

        return counts

    # Get the statevector and display on a Bloch sphere
    elif visual == 'bloch':
        backend = BasicAer.get_backend('statevector_simulator')
        statevector = execute(circ, backend).result().get_statevector()
        get(circ)
        display(plot_bloch_multivector(statevector))

    # Just get counts
    else:
        simulator = Aer.get_backend('qasm_simulator')
        results = execute(circ, simulator).result()
        counts = results.get_counts(circ)

        return counts


def cx_check(test_unitary, perfect=False):
    """
    Check if the CnX unitary is correct

    Parameters:
    -----------
    test_unitary: ndarray
        Unitary generated by the circuit
    perfect: ndarray
        Account for phase difference

    """

    # Get length of unitary

    if not perfect:
        test_unitary = np.abs(test_unitary)

    size = test_unitary.shape[0]

    cx_theory = np.identity(size)

    # Change all the difference
    cx_theory[int(size / 2) - 1, size - 1] = 1
    cx_theory[size - 1, int(size / 2) - 1] = 1
    cx_theory[int(size / 2) - 1, int(size / 2) - 1] = 0
    cx_theory[size - 1, size - 1] = 0

    # Assert Similarity
    aae(cx_theory, test_unitary)

    print('Unitary is correct')


def CnX(n, control_list = None, target = None, circ = None, theta = 1):
    
    # Build New Circuit
    if circ == None:
        circ = q(n+1)
        control_list = list(range(n))
        target = n
    
    # Base Case
    if n == 1:
        
        circ.cx(*control_list, target)
        
        return circ
    
    if n==2:
        circ.ch(control_list[0], target)
        circ.cz(control_list[1], target)
        circ.ch(control_list[0], target)
        
        return circ
    
    if n == 3:
        circ.rcccx(*control_list, target)
        
        return circ
    
    if n==4:
        circ.ch(control_list[0], target)
        circ.p(np.pi/8, control_list[1])
        circ.cx(control_list[1], control_list[2])
        circ.p(-np.pi/8, control_list[2])
        circ.cx(control_list[1], control_list[2])
        circ.p(np.pi/8, control_list[2])
        circ.cx(control_list[2], control_list[3])
        circ.p(-np.pi/8, control_list[3])
        circ.cx(control_list[1], control_list[3])
        circ.p(np.pi/8, control_list[3])
        circ.cx(control_list[2], control_list[3])
        circ.p(-np.pi/8, control_list[3])
        circ.cx(control_list[1], control_list[3])
        circ.p(np.pi/8, control_list[3])
        circ.cx(control_list[3], target)
        circ.p(-np.pi/8, target)
        circ.cx(control_list[1], target)
        circ.p(np.pi/8, target)
        circ.cx(control_list[2], target)
        circ.p(-np.pi/8, target)
        circ.cx(control_list[1], target)
        circ.p(np.pi/8, target)
        circ.cx(control_list[3], target)
        circ.p(-np.pi/8, target)
        circ.cx(control_list[1], target)
        circ.p(np.pi/8, target)
        circ.cx(control_list[2], target)
        circ.p(-np.pi/8, target)
        circ.cx(control_list[1], target)
        circ.p(np.pi/8, target)
        circ.ch(control_list[0], target)
        
        return circ
        
    
    # New Case
    
    # CH
    circ.ch(control_list[0], target)
    
    # CnP
    circ = CnP(n-1, control_list[1:], target, circ, theta)
    
    # CH
    circ.ch(control_list[0], target)
    
    return circ

def CnP(n, control_list = None, target = None, circ = None, theta = 1):
    
    # Build New Circuit
    if circ == None:
        circ = q(n+1)
        control_list = list(range(n))
        target = n
    
    # Base Case
        
    if n == 1:
        circ = control_phase(circ, theta, control_list, target)
        
        return circ 
    
    if n==2:
        circ.p(np.pi/(theta*(2**n)), control_list[0])
        circ.cx(control_list[0], control_list[1])
        circ.p(-np.pi/(theta*2**n), control_list[1])
        circ.cx(control_list[0], control_list[1])
        circ.p(np.pi/(theta*2**n), control_list[1])
        circ.cx(control_list[1], target)
        circ.p(-np.pi/(theta*2**n), target)
        circ.cx(control_list[0], target)
        circ.p(np.pi/(theta*2**n), target)
        circ.cx(control_list[1], target)
        circ.p(-np.pi/(theta*2**n), target)
        circ.cx(control_list[0], target)
        circ.p(np.pi/(theta*2**n), target)

        return circ

    if n==3:
        circ.p(np.pi/(theta*2**n), control_list[0])
        circ.cx(control_list[0], control_list[1])
        circ.p(-np.pi/(theta*2**n), control_list[1])
        circ.cx(control_list[0], control_list[1])
        circ.p(np.pi/(theta*2**n), control_list[1])
        circ.cx(control_list[1], control_list[2])
        circ.p(-np.pi/(theta*2**n), control_list[2])
        circ.cx(control_list[0], control_list[2])
        circ.p(np.pi/(theta*2**n), control_list[2])
        circ.cx(control_list[1], control_list[2])
        circ.p(-np.pi/(theta*2**n), control_list[2])
        circ.cx(control_list[0], control_list[2])
        circ.p(np.pi/(theta*2**n), control_list[2])
        circ.cx(control_list[0], target)
        circ.p(-np.pi/(theta*2**n), target)
        circ.cx(control_list[1], target)
        circ.p(np.pi/(theta*2**n), target)
        circ.cx(control_list[2], target)
        circ.p(-np.pi/(theta*2**n), target)
        circ.cx(control_list[1], target)
        circ.p(np.pi/(theta*2**n), target)
        circ.cx(control_list[0], target)
        circ.p(-np.pi/(theta*2**n), target)
        circ.cx(control_list[1], target)
        circ.p(np.pi/(theta*2**n), target)
        circ.cx(control_list[2], target)
        circ.p(-np.pi/(theta*2**n), target)
        circ.cx(control_list[1], target)
        circ.p(np.pi/(theta*2**n), target)

        return circ
        
    
    # New Case
    
    # CP
    circ = control_phase(circ, theta*2, control_list[-1], target)
    
    # C(n-1)X
    circ = CnX(n-1, control_list[:-1], control_list[-1], circ)
    
    # -CP
    circ = control_phase(circ, -theta*2, control_list[-1], target)
    
    # C(n-1)X
    circ = CnX(n-1, control_list[:-1], control_list[-1], circ)
    
    # C(n-1)P
    circ = CnP(n-1, control_list[:-1], target, circ, theta*2)
    
    return circ


def CnH(n, control_list=None, target=None, circ=None, theta=1):
    # Build New Circuit
    if circ == None:
        circ = q(n + 1)
        control_list = list(range(n))
        target = n

    # Base Case

    if n == 1 and theta == 1:
        circ.ch(control_list, target)

        return circ

    if n == 1:
        circ.unitary(cu_matrix(2, 0, 1, Hp(1 / theta)), [control_list, target])

        return circ

        # New Case

    # CH
    circ.unitary(cu_matrix(2, 0, 1, Hp(1 / (theta * 2))), [control_list[-1], target])

    # C(n-1)X
    circ = CnX(n - 1, control_list[:-1], control_list[-1], circ)

    # CH
    circ.unitary(cu_matrix(2, 0, 1, Hp(-1 / (theta * 2))), [control_list[-1], target])

    # C(n-1)X
    circ = CnX(n - 1, control_list[:-1], control_list[-1], circ)

    # C(n-1)P
    circ = CnH(n - 1, control_list[:-1], target, circ, theta * 2)

    return circ


def h_relief(n, no_h, return_circ=False):
    # n is the number of control qubit
    # no_h is the number of control qubit on the side hadamard
    circ = q(n + 1)
    circ = CnH(no_h, list(range(no_h)), n, circ)

    circ = CnP(n - no_h, list(range(no_h, n)), n, circ)
    circ = CnH(no_h, list(range(no_h)), n, circ)

    '''# Test for accuracy
    test = get(circ, nice = False)
    unitary_check(test)'''

    if return_circ:
        return circ

    dtp(circ, nice=False)
