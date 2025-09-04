"""
Optimized implementation for calculating Bernoulli numbers and detecting irregular primes.
Uses efficient algorithms to avoid the recursive approach of the previous implementation.
"""

import sympy
from fractions import Fraction
from typing import Dict, List, Set, Tuple
import time
from yans4 import YANSNumber, yans_representation

class FastBernoulliCalculator:
    """
    Fast calculator for Bernoulli numbers using efficient algorithms.
    """
    def __init__(self):
        # Initialize with B0 = 1 and B1 = -1/2
        self._cache = {0: Fraction(1), 1: Fraction(-1, 2)}
        # All odd-indexed Bernoulli numbers beyond B1 are zero
        for n in range(3, 100, 2):
            self._cache[n] = Fraction(0)
        
    def get(self, n: int) -> Fraction:
        """
        Get the nth Bernoulli number using the efficient Akiyama-Tanigawa algorithm.
        """
        if n in self._cache:
            return self._cache[n]
        
        # Direct computation for even Bernoulli numbers using the Akiyama-Tanigawa algorithm
        # This is much faster than the recursive approach
        if n % 2 == 0:
            m = n // 2
            arr = [Fraction(1, k+1) for k in range(m+1)]
            
            for j in range(1, m+1):
                for k in range(m, j-1, -1):
                    arr[k] = k * (arr[k] - arr[k-1])
            
            result = arr[m]
            if n > 2:
                # Bernoulli numbers with even indices > 2 alternate in sign
                result = -result
            
            self._cache[n] = result
            return result
        
        # All odd-indexed Bernoulli numbers beyond B1 are zero
        self._cache[n] = Fraction(0)
        return Fraction(0)

class IrregularPrimeDetector:
    """
    Efficient detector for irregular primes using optimized algorithms.
    """
    def __init__(self):
        self._bernoulli = FastBernoulliCalculator()
        self._irregular_primes_cache: Set[int] = set()
        self._regular_primes_cache: Set[int] = set()
        
    def is_irregular(self, p: int) -> bool:
        """
        Check if p is an irregular prime. 
        A prime p is irregular if it divides the numerator of any Bernoulli number B_2k
        where 2k ≤ p-3.
        """
        if not sympy.isprime(p):
            raise ValueError(f"{p} is not a prime")
            
        # Check cache first
        if p in self._irregular_primes_cache:
            return True
        if p in self._regular_primes_cache:
            return False
            
        # Check if p divides any B_2k numerator
        for k in range(1, (p-1)//2):
            idx = 2 * k
            if idx > p - 3:
                break
                
            # Use modular arithmetic for efficiency
            # If B_2k ≡ 0 (mod p), then p divides the numerator
            bernoulli = self._bernoulli.get(idx)
            if bernoulli.numerator % p == 0:
                self._irregular_primes_cache.add(p)
                return True
                
        self._regular_primes_cache.add(p)
        return False
        
    def find_irregular_primes(self, limit: int) -> List[int]:
        """
        Find all irregular primes up to the given limit.
        Returns a list of (prime, [list of indices where p divides B_2k]).
        """
        result = []
        
        for p in sympy.primerange(3, limit + 1):
            indices = []
            is_irregular = False
            
            # Check if p divides any B_2k numerator
            for k in range(1, (p-1)//2):
                idx = 2 * k
                if idx > p - 3:
                    break
                    
                bernoulli = self._bernoulli.get(idx)
                if bernoulli.numerator % p == 0:
                    indices.append(idx)
                    is_irregular = True
            
            if is_irregular:
                result.append((p, indices))
                self._irregular_primes_cache.add(p)
            else:
                self._regular_primes_cache.add(p)
                
        return result

def run_benchmark():
    """
    Run a benchmark to find irregular primes up to 200.
    """
    print("Finding irregular primes up to 200...")
    start_time = time.time()
    
    detector = IrregularPrimeDetector()
    irregular_primes = detector.find_irregular_primes(200)
    
    end_time = time.time()
    print(f"Found {len(irregular_primes)} irregular primes in {end_time - start_time:.2f} seconds:")
    
    for p, indices in irregular_primes:
        bernoulli_str = ", ".join(f"B_{idx}" for idx in indices)
        print(f"  {p} divides {bernoulli_str}")
    
    # List first few values for verification
    calculator = FastBernoulliCalculator()
    print("\nFirst few Bernoulli numbers:")
    for i in range(0, 11):
        b = calculator.get(i)
        print(f"B_{i} = {b}")

if __name__ == "__main__":
    run_benchmark()
