"""Code for running a small instance of VLS on IBMQ."""

# =============================================================================
# imports
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

from qiskit import (register, QuantumRegister, ClassicalRegister,
                    QuantumCircuit, execute, IBMQ)
from scipy.optimize import minimize


# =============================================================================
# constants
# =============================================================================

# api token and url to register to use ibmq
API = ""
URL = ""

# number of qubits in the system
nqubits = 1

# number of times to run hadamard test to estimate observable
NUM_SHOTS = 1000

# flags
VERBOSE = True

# optimization methods
METHODS = ['Nelder-Mead', 'Powell',
           'CG', 'BFGS', 'COBYLA',
           'TNC', 'trust-constr',
           'trust-ncg', 'trust-krylov',
           'trust-exact']

# index for optimization method to use
m = 1

# =============================================================================
# functions
# =============================================================================

def run_hadamard_test(operators:list, angles=[0, 0, 0],
                      num_shots:int=NUM_SHOTS, mode:str="real",
                      verbose=False):
    """Runs Hadamard test to estimate the observable <0|Q_i|0>
    where each Q_i is an operator in 'operators'.
    """
    # operation key : instruction list
    op_dict = {"X": "circ.cx(qreg[0], qreg[1])",
               "H": "circ.ch(qreg[0], qreg[1])",
               "V": "circ.cu3({}, {}, {}, qreg[0], qreg[1])".format(*angles)}
    
    # =========================================================================
    # get a quantum circuit
    # =========================================================================
    
    qreg = QuantumRegister(2)
    creg = ClassicalRegister(1)
    circ = QuantumCircuit(qreg, creg)
    
    # =========================================================================
    # do the hadamard test
    # =========================================================================
    
    # first hadamard on top register
    circ.h(qreg[0])
    
    # all controlled operations
    for op in operators:
        exec(op_dict[op])
    
    # second hadamard on top register
    circ.h(qreg[0])
    
    # measurement basis for real or imag part
    if mode == "im" or mode == "imag" or mode != "real":
        circ.s(qreg[0])
        
    # measure top register
    circ.measure(qreg[0], creg[0])
    
    # =========================================================================
    # run the circuit and get the results
    # =========================================================================
    
    if verbose:
        print(circ.qasm())
    
    # TODO: explore the multiple circuits option of execute()
    # can we get the real and imag parts in one call?
    out = execute(circuits=circ, backend="qasm_simulator", shots=num_shots)
    res = out.result()
    counts = res.get_counts()
    
    # get the counts of 0 and 1
    if "0" in counts.keys():
        zero_count = counts["0"]
    else:
        zero_count = 0

    if "1" in counts.keys():
        one_count = counts["1"]
    else:
        one_count = 0

    return (zero_count - one_count) / num_shots

def compute_expectation(operators:list, angles=[0, 0, 0],
                        num_shots:int=NUM_SHOTS, verbose=False):
    """Computes the expectation of an observable <0|Q|0>.
    
    Here, Q corresponds to an operator list in VLS.
    """
    if verbose:
        print("Now computing real part of expectation value..")
    real = run_hadamard_test(operators, angles, num_shots, "real", verbose)

    if verbose:
        print("Now computing imag part of expectation value...")
    imag = run_hadamard_test(operators, angles, num_shots, "imag", verbose)
    return real + imag * 1j

def cost(angles):
    """Computes the global cost function for the simple 2x2 example with
    A = Hadmard and b = X|0>.
    """
    ops = ["X", "H", "V"]
    expectation = compute_expectation(ops, angles)
    overlap = abs(expectation)**2
    cost = 1 - overlap
    print(cost)
    return cost

def grid_search(step):
    """Runs a grid search to compute cost over all angles."""
    xs = ys = zs = np.arange(0, 2 * np.pi, step)
    
    costs = np.zeros([len(xs), len(ys), len(zs)])
    
    for (i, x) in enumerate(xs):
        for (j, y) in enumerate(ys):
            for (k, z) in enumerate(zs):
                costs[i, j, k] = cost([x, y, z])
    return costs

# =============================================================================
# main
# =============================================================================

def main():
    """Runs main function for the file."""
    init_angles = [0, 0, 0]

    # =========================================================================
    # do the optimization
    # =========================================================================
    
    out = minimize(fun=cost, x0=init_angles,
                   bounds=[(0, 2 * np.pi)] * 3, method=METHODS[m])

    print(out)

# =============================================================================
# main script
# =============================================================================

if __name__ == "__main__":
    main()
