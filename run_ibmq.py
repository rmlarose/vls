"""Code for running a small instance of VLS on IBMQ."""

# =============================================================================
# imports
# =============================================================================

from qiskit import (register, QuantumRegister, ClassicalRegister,
                    QuantumCircuit, execute, IBMQ)

# =============================================================================
# constants
# =============================================================================

# api token and url to register to use ibmq
API = ""
URL = ""

# number of qubits in the system
nqubits = 1

# number of times to run hadamard test to estimate observable
num_shots = 10000

# flags
VERBOSE = True

# =============================================================================
# functions
# =============================================================================

def run_hadamard_test(operators:list, angles=[0, 0, 0],
                      num_shots:int=10000, mode:str="real",
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
    
    # return estimate of expectation
    return (counts["0"] - counts["1"]) / num_shots

def compute_expectation(operators:list, angles=[0, 0, 0],
                        num_shots:int=10000, verbose=False):
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

# =============================================================================
# main
# =============================================================================

def main():
    """Runs main function for the file."""
    ops1 = ["X", "H", "V"]
    ops2 = ["V", "H", "X"]
    
    test_angles = [1, 2, 3]
    
    expec1 = compute_expectation(ops1, test_angles)
    
    overlap = abs(expec1)**2
    cost = 1 - overlap

    print("the cost is", cost)

# =============================================================================
# main script
# =============================================================================

# register to use ibmq
#register(API, URL)

# make classical and quantum registers
qreg = QuantumRegister(nqubits + 1)
creg = ClassicalRegister(1)

# make a quantum circuit
circ = QuantumCircuit(qreg, creg)

# =============================================================================
# psi state preparation for hadamard test
# =============================================================================

# add the unitary ansatz
circ.u2(0.132, 0.132, qreg[1])

# add the A matrix
circ.h(qreg[1])

# add the b vector preparation (dagger)
circ.x(qreg[1])

# =============================================================================
# hadamard test
# =============================================================================

# first hadamard on top register
circ.h(qreg[0])

# controlled observable
circ.cx(qreg[0], qreg[1])

# second hadamard on top register
circ.h(qreg[0])

# measure the first register
circ.measure(qreg[0], creg[0])

out = execute(circ, "qasm_simulator", shots=num_shots)
res = out.result()
counts = res.get_counts()

if VERBOSE:
    print(circ.qasm())

observable = (counts["0"] - counts["1"]) / num_shots

print(observable)
