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

import cirq

# =============================================================================
# classes
# =============================================================================

class PauliSystem():
    """PauliMatrix class.
    
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

    # =========================================================================
    # matrix methods
    # =========================================================================
    
    def matrix(self):
        """Returns a matrix representation of the PauliMatrix."""
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
    # circuit methods
    # =========================================================================
