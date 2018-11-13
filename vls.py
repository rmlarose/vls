"""
vls.py

Class for variational linear systems code.
"""

# =============================================================================
# imports 
# =============================================================================

import numpy as np

import cirq

# =============================================================================
# constants 
# =============================================================================

test_matrix = 1 / np.sqrt(2) * np.array([[1, 1],
                                         [1, -1]])

test_vector = np.array([1, 0])

# =============================================================================
# vls class
# =============================================================================

class VLS():
    """
    VLS class.
    
    Attributes
    
    Methods
    """
    
    def __init__(self, nqubits=1, amat=test_matrix, bvec=test_vector):
        self.nqubits = nqubits
        self.amat = amat
        self.bvec = bvec
        self.xvec = np.zeros_like(bvec)
        
        # check input
        self._check_matrix()
        self._check_vector()
        
        # normalize input if needed
        
    # =========================================================================
    # helper methods for checking input
    # =========================================================================
    
    def _check_matrix(self):
        # matrix must be a numpy array or numpy matrix
        assert isinstance(self.amat, (np.ndarray, np.matrixlib.defmatrix.matrix))
        
        # must be a square matrix
        assert self.amat.shape[0] == self.amat.shape[1]
        
        # make sure matrix size corresponds to number of qubits
        assert self.amat.size == 2**(2 * self.nqubits)

    def _check_vector(self):
        # must be a numpy array
        assert isinstance(self.bvec, np.ndarray)
        
# =============================================================================
# functions
# =============================================================================

def main():
    """Main function for script."""
    vlscirc = VLS(nqubits=1, amat=test_matrix, bvec=test_vector)

# =============================================================================
# main script
# =============================================================================

if __name__ == "__main__":
    main()