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
API = "3ce852634bcc0d3fc6a5af0920aff9ad4be74ec6972f2a8e06b384796d4b28fd933c7386d2e1296ee5b0425afb317f75d44f558ba6ba18cd8fc899a1fe9fcbb8"
URL = "https://quantumexperience.ng.bluemix.net/api"

# backend to use for circuit execution
BACKEND = "qasm_simulator"

# number of qubits in the system
nqubits = 1

# number of times to run hadamard test to estimate observable
NUM_SHOTS = 10000

# flags
VERBOSE = True

# mode of script (optimize or grid search)
MODE = "opt"

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
    out = execute(circuits=circ,
                  backend=BACKEND,
                  shots=num_shots)
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
    expectation = compute_expectation(ops, angles + [0, 0])
    overlap = abs(expectation)**2
    cost = 1 - overlap
    print(cost)
    return cost

def grid_search(step):
    """Runs a grid search to compute cost over all angles."""
    xs = np.arange(0, 2 * np.pi, step)

    costs = np.zeros_like(xs)

    for (i, x) in enumerate(xs):
        costs[i] = cost([x])
    return (xs, costs)

# =============================================================================
# main
# =============================================================================

def main():
    """Runs main function for the file."""
    # register to use ibmq
    IBMQ.enable_account(API, URL)

    # =========================================================================
    # grid search mode
    # =========================================================================

    if MODE == "grid_search":
        xs, costs = grid_search(step=0.1)
        plt.plot(xs, costs, "-o", linewidth=2)
        plt.xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                   ["0", "pi / 2", "pi", "3 pi / 2", "2 pi"])
        plt.grid()
        plt.show()

    # =========================================================================
    # optimization mode
    # =========================================================================
    
    else:
        init_angles = [np.pi / 2]
        """
        out = minimize(fun=cost,
                       x0=init_angles,
                       bounds=[(0, 2 * np.pi)] * len(init_angles),
                       method=METHODS[m])
        """
        out = cost(init_angles)

        print(out)

# =============================================================================
# main script
# =============================================================================

if __name__ == "__main__":
    main()
