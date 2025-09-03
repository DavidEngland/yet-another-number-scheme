"""
Implementation of Gelfond-Schneider Theorem for YANS.

The Gelfond-Schneider Theorem states that if α and β are algebraic numbers 
with α ≠ 0, α ≠ 1, and β irrational, then α^β is transcendental.
"""

from typing import Tuple, Optional, List, Dict, Union
import sympy
from yans3 import YANSNumber, yans_representation

class AlgebraicNumber:
    """
    Represents an algebraic number (root of a polynomial with rational coefficients).
    
    For simple algebraic numbers, we store:
    - Minimal polynomial as a list of coefficients
    - Root index if multiple roots exist
    - YANS representation for rational numbers
    """
    def __init__(self, 
                 minimal_poly: List[int], 
                 root_index: int = 0,
                 yans_value: Optional[YANSNumber] = None):
        """
        Initialize an algebraic number.
        
        Args:
            minimal_poly: Coefficients of minimal polynomial, highest degree first
            root_index: Which root of the polynomial (0-indexed)
            yans_value: Exact YANS representation if rational
        """
        self.minimal_poly = minimal_poly
        self.root_index = root_index
        self.yans_value = yans_value
        self._sympy_poly = None
        self._value = None
    
    @classmethod
    def from_rational(cls, yans: YANSNumber) -> 'AlgebraicNumber':
        """Create an algebraic number from a rational number."""
        n = yans.to_int()
        # Minimal polynomial: x - n = 0
        return cls([-n, 1], 0, yans)
    
    @classmethod
    def sqrt(cls, n: int) -> 'AlgebraicNumber':
        """Create an algebraic number representing √n."""
        # Minimal polynomial: x^2 - n = 0
        return cls([-n, 0, 1], 0)
    
    @property
    def is_rational(self) -> bool:
        """Check if this algebraic number is rational."""
        return self.yans_value is not None or len(self.minimal_poly) == 2
    
    @property
    def is_irrational(self) -> bool:
        """Check if this algebraic number is irrational."""
        return not self.is_rational
    
    @property
    def sympy_poly(self):
        """Get the sympy polynomial for numerical computation."""
        if self._sympy_poly is None:
            x = sympy.Symbol('x')
            # Convert coefficients to polynomial (high degree to low)
            terms = [c * x**(len(self.minimal_poly)-i-1) 
                     for i, c in enumerate(self.minimal_poly)]
            self._sympy_poly = sum(terms)
        return self._sympy_poly
    
    @property
    def value(self) -> float:
        """Compute numerical approximation of this algebraic number."""
        if self._value is None:
            if self.yans_value is not None:
                self._value = self.yans_value.to_int()
            else:
                # Find all roots numerically
                roots = sympy.nroots(self.sympy_poly)
                if self.root_index < len(roots):
                    self._value = float(roots[self.root_index])
                else:
                    raise ValueError(f"Root index {self.root_index} exceeds "
                                    f"number of roots {len(roots)}")
        return self._value
    
    def __str__(self) -> str:
        if self.yans_value is not None:
            return f"AlgNum({self.yans_value})"
        return f"AlgNum(poly={self.minimal_poly}, root={self.root_index})"
    
    def is_equal_to(self, other: 'AlgebraicNumber', tolerance: float = 1e-10) -> bool:
        """Check if two algebraic numbers are equal (approximately)."""
        return abs(self.value - other.value) < tolerance


def is_transcendental_by_gelfond_schneider(base: AlgebraicNumber, 
                                          exponent: AlgebraicNumber) -> bool:
    """
    Determine if base^exponent is transcendental using Gelfond-Schneider Theorem.
    
    Returns True if base^exponent is definitely transcendental by the theorem,
    False if undetermined (could be algebraic or transcendental).
    """
    # Gelfond-Schneider: If α and β are algebraic with α ≠ 0, α ≠ 1,
    # and β irrational, then α^β is transcendental
    
    # Check base conditions
    if base.is_equal_to(AlgebraicNumber.from_rational(yans_representation(0))):
        return False  # 0^anything is 0 (algebraic)
    
    if base.is_equal_to(AlgebraicNumber.from_rational(yans_representation(1))):
        return False  # 1^anything is 1 (algebraic)
    
    # If exponent is irrational and base meets conditions, result is transcendental
    if exponent.is_irrational:
        return True
    
    # Otherwise, undetermined by this theorem
    return False


def compute_power(base: AlgebraicNumber, exponent: AlgebraicNumber) -> Union[AlgebraicNumber, str]:
    """
    Compute base^exponent, returning either:
    - An AlgebraicNumber if result is provably algebraic
    - A string description if result is transcendental by Gelfond-Schneider
    - A numerical approximation otherwise
    """
    # Check if result is definitely transcendental
    if is_transcendental_by_gelfond_schneider(base, exponent):
        return f"Transcendental by Gelfond-Schneider: {base}^{exponent}"
    
    # Handle simple rational cases
    if base.is_rational and exponent.is_rational:
        if exponent.yans_value is not None:
            # Integer exponent
            exp_val = exponent.yans_value.to_int()
            if int(exp_val) == exp_val:
                # Simple integer power of rational
                if base.yans_value is not None:
                    return AlgebraicNumber.from_rational(base.yans_value ** int(exp_val))
    
    # Return numeric approximation for other cases
    return f"≈ {base.value ** exponent.value}"


# Example usage functions
def sqrt2_demo():
    """Demonstrate √2 is algebraic but √2^√2 is transcendental."""
    sqrt2 = AlgebraicNumber.sqrt(2)
    print(f"√2 is {'irrational' if sqrt2.is_irrational else 'rational'}")
    print(f"√2 ≈ {sqrt2.value}")
    
    # Check if √2^√2 is transcendental
    result = is_transcendental_by_gelfond_schneider(sqrt2, sqrt2)
    print(f"Is √2^√2 transcendental by Gelfond-Schneider? {result}")
    print(f"√2^√2 {compute_power(sqrt2, sqrt2)}")
    
    # Check if √2^2 = 2 is algebraic
    two = AlgebraicNumber.from_rational(yans_representation(2))
    result = is_transcendental_by_gelfond_schneider(sqrt2, two)
    print(f"Is √2^2 transcendental? {result}")
    print(f"√2^2 = {compute_power(sqrt2, two)}")


if __name__ == "__main__":
    sqrt2_demo()
