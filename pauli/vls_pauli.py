"""
Code for running VLS where the matrix A is assumed to be a sum of paulis.

written by Ryan LaRose <rlarose@umich.edu>
at LANL 11-14-2018
"""

# =============================================================================
# imports
# =============================================================================

import numpy as np

import cirq

# =============================================================================
# constants
# =============================================================================

Amat_ops = np.array([["X", "Z", "Z", "I"],
                     ["I", "I", "X", "Z"],
                     ["Z", "I", "I", "Z"]])

Amat_coeffs = np.array([1., -1, 2.])

class PauliMatrix():
    def __init__(self, coeffs, ops):
        # check inputs
        assert len(coeffs) == ops.shape[0]
        
        # attributes
        self.coeffs = coeffs
        self.ops = ops
    
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
    
    def matrix(self):
        """Returns a matrix representation of the PauliMatrix."""
        pass

