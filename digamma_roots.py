"""
Efficient calculation and storage of Gauss digamma function roots.

The digamma function ψ(z) = d/dz[ln(Γ(z))] = Γ'(z)/Γ(z) has roots
that are important in number theory and special function applications.
"""

import mpmath
import numpy as np
import sympy
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
import pickle
import os

from yans3 import YANSNumber, yans_representation

# Set precision for mpmath
mpmath.mp.dps = 50  # 50 digits of precision

class DigammaRoots:
    """
    Class for calculating and storing digamma function roots efficiently.
    Uses a combination of mpmath for high-precision calculation and 
    caching mechanisms for performance.
    """
    def __init__(self, cache_file: Optional[str] = None):
        self.roots_cache: Dict[int, mpmath.mpf] = {}
        self.cache_file = cache_file or os.path.join(
            os.path.dirname(__file__), 'digamma_roots_cache.pkl'
        )
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cached roots from disk if available."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    # We store as strings for better serialization
                    string_cache = pickle.load(f)
                    self.roots_cache = {
                        int(k): mpmath.mpf(v) for k, v in string_cache.items()
                    }
            except (pickle.PickleError, IOError):
                # If loading fails, start with empty cache
                self.roots_cache = {}
    
    def _save_cache(self) -> None:
        """Save cached roots to disk."""
        try:
            # Convert mpmath.mpf to strings for reliable serialization
            string_cache = {
                str(k): str(v) for k, v in self.roots_cache.items()
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(string_cache, f)
        except (pickle.PickleError, IOError):
            # Silently fail if we can't save the cache
            pass
    
    @staticmethod
    @lru_cache(maxsize=128)
    def digamma(z: complex) -> complex:
        """
        Calculate the digamma function at z.
        Uses mpmath for high precision.
        """
        return complex(mpmath.psi(0, complex(z)))
    
    def find_root(self, n: int) -> mpmath.mpf:
        """
        Find the nth negative root of the digamma function.
        
        The digamma function has roots at negative non-integers.
        n=1 corresponds to the root between -1 and 0,
        n=2 corresponds to the root between -2 and -1, etc.
        
        Uses cache for repeated calls.
        """
        # Check cache first
        if n in self.roots_cache:
            return self.roots_cache[n]
        
        # Initial guess: roots are close to negative integers
        x0 = -n + 0.5
        
        # Use mpmath's findroot for high precision
        try:
            root = mpmath.findroot(lambda x: mpmath.psi(0, x), x0)
            # Store in cache
            self.roots_cache[n] = root
            self._save_cache()
            return root
        except ValueError:
            # Fallback with different initial guess if first attempt fails
            x0 = -n + 0.4
            root = mpmath.findroot(lambda x: mpmath.psi(0, x), x0)
            self.roots_cache[n] = root
            self._save_cache()
            return root
    
    def get_roots(self, count: int) -> List[mpmath.mpf]:
        """
        Get the first 'count' negative roots of the digamma function.
        """
        return [self.find_root(n) for n in range(1, count+1)]
    
    def to_yans(self, n: int) -> YANSNumber:
        """
        Convert the nth digamma root to its best rational approximation
        and return as a YANSNumber.
        """
        root = self.find_root(n)
        # Convert to fraction for rational approximation
        frac = mpmath.nstr(root, n=20, min_fixed=-1, max_fixed=-1)
        # Parse the string representation
        sym_frac = sympy.sympify(frac)
        # Convert to YANS
        return yans_representation(int(sym_frac))
    
    def as_continued_fraction(self, n: int, terms: int = 10) -> List[int]:
        """
        Represent the nth digamma root as a continued fraction.
        Returns the first 'terms' coefficients.
        """
        root = self.find_root(n)
        return mpmath.nstr(mpmath.continued_fraction(root, terms), n=terms)

# Convenience function to get a digamma root calculator
def get_digamma_roots() -> DigammaRoots:
    """Get a singleton instance of DigammaRoots."""
    if not hasattr(get_digamma_roots, '_instance'):
        get_digamma_roots._instance = DigammaRoots()
    return get_digamma_roots._instance

# Example function to demonstrate usage
def digamma_root_demo() -> None:
    """Demonstrate finding and working with digamma roots."""
    roots = get_digamma_roots()
    
    print("First 5 roots of the digamma function:")
    for i in range(1, 6):
        root = roots.find_root(i)
        print(f"Root #{i}: {root}")
        print(f"Verification: digamma({root}) = {mpmath.psi(0, root)}")
        print(f"Continued fraction: {roots.as_continued_fraction(i, 5)}")
        print()
    
    # Show how to convert to YANS representation
    print("YANS representation of first digamma root:")
    yans_root = roots.to_yans(1)
    print(f"As YANS: {yans_root}")
    print(f"As factor string: {yans_root.to_factor_string()}")

if __name__ == "__main__":
    digamma_root_demo()
