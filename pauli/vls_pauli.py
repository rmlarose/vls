"""
Classes and functions for running VLS where the matrix A is assumed to be a 
sum of pauli operators.

written by Ryan LaRose <rlarose@umich.edu>
at LANL 11-14-2018
"""

# =============================================================================
# imports
# =============================================================================

import numpy as np

from cirq import (Circuit, InsertStrategy, LineQubit, ops)
from cirq.ops.controlled_gate import ControlledGate

# =============================================================================
# classes
# =============================================================================

class PauliSystem():
    """PauliSystem class.
    
    """
    # =========================================================================
    # matrices of pauli operators
    # =========================================================================
    
    # identity
    imat = np.array([[1., 0.],
                     [0., 1.]], dtype=np.complex64)

    # pauli x
    xmat = np.array([[0., 1.],
                     [1., 0.]], dtype=np.complex64)

    # pauli y
    ymat = np.array([[0., -1j],
                     [1j, 0.]], dtype=np.complex64)

    # pauli z
    zmat = np.array([[1., 0.],
                     [0., -1.]], dtype=np.complex64)

    # =========================================================================
    # basic methods
    # =========================================================================

    def __init__(self, coeffs, ops, vec_ops):
        # check inputs
        assert len(coeffs) == ops.shape[0]
        assert len(vec_ops) == ops.shape[1]

        # attributes
        self.coeffs = coeffs
        self.ops = ops
        self.vec_ops = vec_ops

    def num_qubits(self):
        """Returns the number of qubits the PauliMatrix acts on."""
        return self.ops.shape[1]
    
    def size(self):
        """Returns the dimensions of the PauliMatrix."""
        dim = 2**(self.num_qubits())
        return (dim, dim)

    def num_elements(self):
        """Returns the number of elements in the PauliMatrix."""
        return 2**(2 * self.num_qubits())

    # =========================================================================
    # methods for the matrix of the Pauli System
    # =========================================================================
    
    def matrix(self):
        """Returns a matrix representation of the matrix of the PauliSystem."""
        # allocate space
        mat = np.zeros(self.size(), dtype=np.complex64)
        
        # loop over all terms in operators
        for (ind, op_list) in enumerate(self.ops):
            
            # form the first kronecker product
            if len(op_list) == 1:
                term = self._key_to_mat(op_list[0])
            else:
                term = np.kron(self._key_to_mat(op_list[0]),
                               self._key_to_mat(op_list[1]))
            
            # form all other terms in kronecker product
            for k in range(2, len(op_list)):
                key = op_list[k]
                term = np.kron(term, self._key_to_mat(key))
            
            # add the term to the matrix
            mat += self.coeffs[ind] * term
            
        return mat
            
            
    def _key_to_mat(self, key:str):
        """Returns a pauli matrix corresponding to a string key."""
        key_mat = {"I": self.imat,
                   "X": self.xmat,
                   "Y": self.ymat,
                   "Z": self.zmat}
        
        return key_mat[key]

    # =========================================================================
    # methods for creating circuits
    # =========================================================================
    
    def _key_to_gate(self, key:str):
        """Returns a gate corresponding to a string key."""
        key_mat = {"X": ops.X,
                   "Y": ops.Y,
                   "Z": ops.Z}
        
        return key_mat[key]
    
    def _key_to_cgate(self, key:str):
        """Returns a controlled-gate corresponding to a string key."""
        key_mat = {"X": ControlledGate(ops.X),
                   "Y": ControlledGate(ops.Y),
                   "Z": ControlledGate(ops.Z)}
        
        return key_mat[key]
    
    
    def make_matrix_circuit(self):
        """Returns a quantum circuit implementing the matrix of the
        PauliSystem.
        """
        # get a circuit
        circ = Circuit()
        
        # get some qubits
        qbits = [LineQubit(x) for x in range(self.num_qubits() + 1)]
        
        # loop over each term in the matrix expansion
        for op_list in self.ops:
            # loop over each pauli operator
            for (q, key) in enumerate(op_list):
                if key == "I": continue
                q += 1
                circ.append(
                    self._key_to_gate(key)(qbits[q]),
                    strategy=InsertStrategy.EARLIEST
                    )
        
        return circ
                
    
    def make_controlled_matrix_circuit(self):
        """Returns a quantum circuit implementing the matrix of the
        PauliSystem.
        """
        # get a circuit
        circ = Circuit()
        
        # get some qubits
        qbits = [LineQubit(x) for x in range(self.num_qubits() + 1)]
        
        # loop over each term in the matrix expansion
        for op_list in self.ops:
            # loop over each pauli operator
            for (q, key) in enumerate(op_list):
                q += 1
                if key == "I": continue
                circ.append(
                    self._key_to_cgate(key)(qbits[0], qbits[q]),
                    strategy=InsertStrategy.EARLIEST
                    )
        
        return circ
    
    def make_vector_circuit(self):
        """Returns a quantum circuit implementing the unitary U that prepares
        the solution vector b from the ground state.
        
        That is, |b> = U|0>.
        """
        # get a circuit
        circ = Circuit()
        
        # get some qubits
        qbits = [LineQubit(x) for x in range(self.num_qubits() + 1)]
        
        # loop over each pauli operator
        for (q, key) in enumerate(self.vec_ops):
            if key == "I": continue
            q += 1
            circ.append(
                self._key_to_gate(key)(qbits[q]),
                strategy=InsertStrategy.EARLIEST
                )
        
        return circ

    def make_controlled_vector_circuit(self):
        """Returns a quantum circuit implementing the controlled unitary U 
        that prepares the solution vector b from the ground state.
        
        That is, |b> = U|0>.
        """
        # get a circuit
        circ = Circuit()
        
        # get some qubits
        qbits = [LineQubit(x) for x in range(self.num_qubits() + 1)]
        
        # loop over each pauli operator
        for (q, key) in enumerate(self.vec_ops):
            if key == "I": continue
            q += 1
            circ.append(
                self._key_to_cgate(key)(qbits[0], qbits[q]),
                strategy=InsertStrategy.EARLIEST
                )
        
        return circ
    
    def make_ansatz_circuit(self):
        pass
    
    def _make_hadamard_test_circuit(self, ops1, ops2, j, mode):
        # add hadamard gate on top register
        
        # add ansatz on bottom register
        
        # add controlled sigma_k term (corresponding to ops1)
        
        # add u dagger term
        
        # add controlled sigma_z on the jth qubit
        
        # add u term
        
        # add controlled sigma_kprim term (corresponding to ops2)
        
        # optional s gate for imag part
        
        # add hadamard gate on top register
        
        # add measurement to top qubit
        pass

    # =========================================================================
    # methods for computing the cost
    # =========================================================================
    
    
    

    def run_hadamard_test(self, ops1, ops2, j, mode):
        """Returns the real or imaginary part of the term
        
        <Q_{k, kprime}^{j}> := <0|V\dag A_k\dag U P_j U\dag A_kprime V|0\>
        
        using the Hadmard test.
        
        Args:
            ops1 [type: list<str>]
                first list of pauli operators (\sigma_k term in notes)
            
            ops2 [type: list<str>]
                second list of pauli operators (\sigma_kprime in notes)
            
            j [type: int]
                which qubit to perform controlled-Z on (between 1 and n)
            
            mode [type: str]
                "real" or "imag"
                "real" mode computes real part of <Q_{k, kprime}^{j}>
                
                "imag" mode computes imag part of <Q_{k, kprime}^{j}>
        
        Returns:
            rtype: complex
            real/imag part of expectation value <Q_{k, kprime}^{j}>
        """
        # get a hadmard test circuit
        
        # run it
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    