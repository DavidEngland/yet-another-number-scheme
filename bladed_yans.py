"""
YANS implementation extended with geometric algebra concepts (blades).
This file provides a dictionary-based representation of Clifford algebra
elements, with proper geometric product operations.
"""

import sympy
import math
from typing import Dict, Set, List, Tuple, Optional, Union
from yans3 import YANSNumber, yans_representation

def blade_grade(blade: str) -> int:
    """
    Returns the grade of a blade (number of basis vectors).
    Examples: 
      '1' -> 0 (scalar)
      'e1' -> 1 (vector)
      'e12' -> 2 (bivector)
    """
    if blade == '1':
        return 0
    return len(blade) - 1  # Subtract 1 for the 'e' prefix

def geometric_product(blade1: str, blade2: str) -> Tuple[str, int]:
    """
    Compute the geometric product of two blades, returning the resulting blade and sign.
    The sign accounts for the anticommutativity of basis vectors (e_i e_j = -e_j e_i for i ≠ j).
    """
    if blade1 == '1':
        return blade2, 1
    if blade2 == '1':
        return blade1, 1
    
    # Extract indices from blades
    indices1 = [int(i) for i in blade1[1:]]
    indices2 = [int(i) for i in blade2[1:]]
    
    # Compute the geometric product
    result_indices = []
    sign = 1
    
    # First, handle the dot product part (contractions)
    # For each shared index, we get a sign change and the index is removed
    common = set(indices1).intersection(set(indices2))
    for idx in common:
        sign *= -1
    
    # Remaining indices form the outer product
    remaining1 = [i for i in indices1 if i not in common]
    remaining2 = [i for i in indices2 if i not in common]
    result_indices = remaining1 + remaining2
    
    # Sort indices to get canonical form
    result_indices.sort()
    
    # Count the number of swaps needed to sort
    # Each swap introduces a sign change
    swaps = 0
    for i in range(len(result_indices)):
        for j in range(i+1, len(result_indices)):
            if result_indices[i] > result_indices[j]:
                swaps += 1
    
    if swaps % 2 == 1:
        sign *= -1
    
    # Construct result blade
    if not result_indices:
        return '1', sign
    return f"e{''.join(str(i) for i in result_indices)}", sign

class Multivector:
    """
    Represents a general multivector in Clifford algebra using blades and YANS coefficients.
    """
    def __init__(self, elements: Optional[Dict[str, YANSNumber]] = None):
        self.elements = elements or {}  # Maps blade names to YANSNumber coefficients
    
    @classmethod
    def scalar(cls, n: int) -> 'Multivector':
        """
        Creates a scalar multivector.
        """
        return cls({'1': yans_representation(n)})
    
    @classmethod
    def basis_vector(cls, index: int) -> 'Multivector':
        """
        Creates a basis vector (e.g., e1, e2, e3, etc.)
        """
        return cls({f'e{index}': yans_representation(1)})
    
    @classmethod
    def basis_blade(cls, indices: List[int]) -> 'Multivector':
        """
        Creates a basis blade from a list of indices (e.g., [1,2] -> e12)
        """
        if not indices:
            return cls.scalar(1)
        indices.sort()
        blade = f"e{''.join(str(i) for i in indices)}"
        return cls({blade: yans_representation(1)})
    
    @classmethod
    def exp(cls, bivector: 'Multivector') -> 'Multivector':
        """
        Exponential of a bivector, representing a rotor (rotation operator).
        For a bivector B, exp(B) = cos(|B|) + sin(|B|)*B/|B|
        """
        # Only works properly for simple bivectors
        # This is a simplified implementation
        result = {}
        
        # Extract the bivector coefficient
        b_norm = 0
        b_blade = ''
        for blade, coeff in bivector.elements.items():
            if blade_grade(blade) == 2:
                b_norm = coeff.to_int()
                b_blade = blade
        
        # Compute the exponential using the simplified formula for bivectors
        cos_part = yans_representation(math.cos(b_norm))
        sin_part = yans_representation(math.sin(b_norm))
        
        result['1'] = cos_part
        if b_blade in bivector.elements:
            result[b_blade] = sin_part
        
        return cls(result)
    
    def __str__(self) -> str:
        """
        String representation of the multivector.
        """
        terms = []
        for blade, coeff in self.elements.items():
            if coeff.to_int() != 0:
                if blade == '1':
                    terms.append(str(coeff))
                else:
                    terms.append(f"{str(coeff)}{blade}")
        return " + ".join(terms) if terms else "0"
    
    def __add__(self, other: 'Multivector') -> 'Multivector':
        """
        Addition of multivectors.
        """
        result = dict(self.elements)
        for blade, coeff in other.elements.items():
            if blade in result:
                result[blade] = yans_representation(result[blade].to_int() + coeff.to_int())
                if result[blade].to_int() == 0:
                    del result[blade]
            else:
                result[blade] = coeff
        return Multivector(result)
    
    def __sub__(self, other: 'Multivector') -> 'Multivector':
        """
        Subtraction of multivectors.
        """
        result = dict(self.elements)
        for blade, coeff in other.elements.items():
            if blade in result:
                result[blade] = yans_representation(result[blade].to_int() - coeff.to_int())
                if result[blade].to_int() == 0:
                    del result[blade]
            else:
                result[blade] = yans_representation(-coeff.to_int())
        return Multivector(result)
    
    def __mul__(self, other: Union['Multivector', YANSNumber, int]) -> 'Multivector':
        """
        Geometric product of multivectors.
        """
        # Handle scalar multiplication
        if isinstance(other, (YANSNumber, int)):
            scalar = other if isinstance(other, YANSNumber) else yans_representation(other)
            result = {}
            for blade, coeff in self.elements.items():
                result[blade] = coeff * scalar
            return Multivector(result)
        
        # Handle multivector multiplication
        result = {}
        for blade1, coeff1 in self.elements.items():
            for blade2, coeff2 in other.elements.items():
                product_blade, sign = geometric_product(blade1, blade2)
                sign_yans = yans_representation(sign)
                product_coeff = coeff1 * coeff2 * sign_yans
                
                if product_blade in result:
                    result[product_blade] = yans_representation(
                        result[product_blade].to_int() + product_coeff.to_int()
                    )
                    if result[product_blade].to_int() == 0:
                        del result[product_blade]
                else:
                    result[product_blade] = product_coeff
        
        return Multivector(result)
    
    def grade(self, n: int) -> 'Multivector':
        """
        Extract the grade-n part of the multivector.
        """
        return Multivector({
            blade: coeff for blade, coeff in self.elements.items() 
            if blade_grade(blade) == n
        })
    
    def reverse(self) -> 'Multivector':
        """
        Clifford conjugate (reverse the order of basis vectors).
        """
        result = {}
        for blade, coeff in self.elements.items():
            g = blade_grade(blade)
            # Grade 0, 1: unchanged
            # Grade 2, 3: sign flips for odd permutations
            sign = 1 if (g * (g-1)) // 2 % 2 == 0 else -1
            result[blade] = coeff * yans_representation(sign)
        return Multivector(result)
    
    def dual(self, dim: int = 3) -> 'Multivector':
        """
        Compute the dual with respect to the pseudoscalar.
        """
        # Create the pseudoscalar for dimension dim
        pseudoscalar = Multivector.basis_blade(list(range(1, dim+1)))
        # Compute dual by right multiplication with inverse pseudoscalar
        return self * pseudoscalar.reverse()
    
    def norm_squared(self) -> int:
        """
        Compute the squared norm of the multivector.
        """
        # A simple implementation for demonstration
        # Proper implementation would use the scalar part of M*~M
        total = 0
        for _, coeff in self.elements.items():
            val = coeff.to_int()
            total += val * val
        return total

# Helper functions
def outer_product(a: Multivector, b: Multivector) -> Multivector:
    """
    Outer (wedge) product of multivectors.
    """
    result = {}
    for blade1, coeff1 in a.elements.items():
        for blade2, coeff2 in b.elements.items():
            # Only keep terms where indices don't overlap
            indices1 = set() if blade1 == '1' else {int(i) for i in blade1[1:]}
            indices2 = set() if blade2 == '1' else {int(i) for i in blade2[1:]}
            
            if not indices1.intersection(indices2):
                # Compute outer product (essentially a restricted geometric product)
                product_blade, sign = geometric_product(blade1, blade2)
                sign_yans = yans_representation(sign)
                product_coeff = coeff1 * coeff2 * sign_yans
                
                if product_blade in result:
                    result[product_blade] = yans_representation(
                        result[product_blade].to_int() + product_coeff.to_int()
                    )
                else:
                    result[product_blade] = product_coeff
    
    return Multivector(result)

def inner_product(a: Multivector, b: Multivector) -> Multivector:
    """
    Inner (dot) product of multivectors.
    """
    # A simplified implementation that works for vectors
    a_grade1 = a.grade(1)
    b_grade1 = b.grade(1)
    
    result = {}
    for blade1, coeff1 in a_grade1.elements.items():
        for blade2, coeff2 in b_grade1.elements.items():
            # Extract indices
            i = int(blade1[1:])
            j = int(blade2[1:])
            
            # For vectors, inner product is only non-zero when indices match
            if i == j:
                scalar_val = coeff1.to_int() * coeff2.to_int()
                if '1' in result:
                    result['1'] = yans_representation(result['1'].to_int() + scalar_val)
                else:
                    result['1'] = yans_representation(scalar_val)
    
    return Multivector(result)

# Convenience aliases
def yans(n: int) -> YANSNumber:
    """Shorthand for yans_representation"""
    return yans_representation(n)
