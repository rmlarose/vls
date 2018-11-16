"""
Unit tests for PauliSystem.
"""

# =============================================================================
# imports
# =============================================================================

import numpy as np

from vls_pauli import PauliSystem

# =============================================================================
# constants
# =============================================================================

SEP = 40
KEY = "."

# =============================================================================
# unit tests
# =============================================================================

def test_zero_cost(tol=1e-3):
    """Constructs a system with known solution and verifies the cost is
    zero at the correct solution.
    """
    # matrix operators and coefficients
    Amat_ops = np.array([["X", "X", "Z", "Z"]])
    Amat_coeffs = np.array([1-0j])
    
    # vector unitary operators
    bvec_ops = np.array(["X", "X", "Z", "Z"])
    
    # make a system
    system = PauliSystem(Amat_coeffs, Amat_ops, bvec_ops)
    
    assert abs(system.eff_cost(np.zeros(2)) - 0) < tol
    


# =============================================================================
# helper functions
# =============================================================================
    
def passed(test):
    """Prints outt that the test has passed."""
    print(test + "passed!".rjust(40, KEY))

# =============================================================================
# run the tests
# =============================================================================

if __name__ == "__main__":
    test_zero_cost()
    passed("test_zero_cost()")