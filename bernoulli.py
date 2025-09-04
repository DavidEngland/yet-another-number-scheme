"""
Computing and storing Bernoulli numbers using YANS, and classifying
regular vs irregular primes.

Bernoulli numbers are rational numbers that appear in number theory,
and their properties can be used to classify primes.
"""

from fractions import Fraction
from typing import Dict, List, Tuple, Optional
import sympy
from yans4 import YANSNumber, yans_representation

class YANSRational:
    """
    Represents a rational number (fraction) using YANS for both numerator and denominator.
    """
    def __init__(self, numerator: YANSNumber, denominator: YANSNumber):
        """Initialize with numerator and denominator in YANS format."""
        if denominator.to_int() == 0:
            raise ZeroDivisionError("Denominator cannot be zero")
        
        # Simplify the fraction
        self.numerator = numerator
        self.denominator = denominator
        self._simplify()
    
    @classmethod
    def from_fraction(cls, frac: Fraction) -> 'YANSRational':
        """Create a YANSRational from a Python Fraction."""
        num = yans_representation(frac.numerator)
        den = yans_representation(frac.denominator)
        return cls(num, den)
    
    @classmethod
    def from_int(cls, n: int) -> 'YANSRational':
        """Create a YANSRational from an integer."""
        return cls(yans_representation(n), yans_representation(1))
    
    def to_fraction(self) -> Fraction:
        """Convert to Python Fraction."""
        return Fraction(self.numerator.to_int(), self.denominator.to_int())
    
    def _simplify(self) -> None:
        """Simplify the fraction using GCD."""
        num_int = self.numerator.to_int()
        den_int = self.denominator.to_int()
        
        if num_int == 0:
            self.denominator = yans_representation(1)
            return
        
        # Compute GCD and simplify
        gcd = sympy.gcd(num_int, den_int)
        if gcd > 1:
            self.numerator = yans_representation(num_int // gcd)
            self.denominator = yans_representation(den_int // gcd)
        
        # Ensure denominator is positive
        if self.denominator.to_int() < 0:
            self.numerator = yans_representation(-self.numerator.to_int())
            self.denominator = yans_representation(-self.denominator.to_int())
    
    def __str__(self) -> str:
        if self.denominator.to_int() == 1:
            return str(self.numerator)
        return f"{self.numerator}/{self.denominator}"
    
    def __mul__(self, other: 'YANSRational') -> 'YANSRational':
        """Multiply two rational numbers."""
        return YANSRational(
            self.numerator * other.numerator,
            self.denominator * other.denominator
        )


class BernoulliNumbers:
    """
    Generate and store Bernoulli numbers using YANS.
    """
    def __init__(self):
        """Initialize with precomputed first few Bernoulli numbers."""
        self._cache: Dict[int, YANSRational] = {
            0: YANSRational.from_int(1),
            1: YANSRational.from_fraction(Fraction(1, 2)),
        }
        # B_2 = 1/6
        self._cache[2] = YANSRational.from_fraction(Fraction(1, 6))
        # All odd-indexed Bernoulli numbers beyond B_1 are zero
        for i in range(3, 100, 2):
            self._cache[i] = YANSRational.from_int(0)
    
    def get(self, n: int) -> YANSRational:
        """
        Get the nth Bernoulli number.
        Uses a recursive formula and caching for efficiency.
        """
        if n in self._cache:
            return self._cache[n]
        
        # Use recursive formula to compute B_n:
        # B_n = -1/(n+1) * sum(binomial(n+1,k) * B_k for k in range(n))
        result = Fraction(0)
        for k in range(n):
            coef = sympy.binomial(n + 1, k)
            b_k = self.get(k).to_fraction()
            result -= coef * b_k
        
        result /= (n + 1)
        self._cache[n] = YANSRational.from_fraction(result)
        return self._cache[n]

    def classify_prime(self, p: int) -> bool:
        """
        Classify if a prime p is regular or irregular.
        Returns True if p is regular, False if irregular.
        
        A prime p is irregular if it divides the numerator of B_2k
        for some even k where 2 ≤ 2k ≤ p-3.
        """
        if not sympy.isprime(p):
            raise ValueError(f"{p} is not a prime number")
        
        # Check divisibility of B_2k numerators for 2 ≤ 2k ≤ p-3
        for k in range(1, (p - 1) // 2):
            idx = 2 * k
            if idx > p - 3:
                break
                
            bernoulli = self.get(idx)
            # Check if p divides numerator
            if bernoulli.numerator.to_int() % p == 0:
                return False  # Irregular
        
        return True  # Regular

    def analyze_irregularity(self, p: int) -> List[Tuple[int, int]]:
        """
        For an irregular prime, find all pairs (2k, p) where p divides the 
        numerator of B_2k, with 2 ≤ 2k ≤ p-3.
        
        Returns a list of (index, prime divisor) pairs.
        """
        if not sympy.isprime(p):
            raise ValueError(f"{p} is not a prime number")
        
        irregular_pairs = []
        for k in range(1, (p - 1) // 2):
            idx = 2 * k
            if idx > p - 3:
                break
                
            bernoulli = self.get(idx)
            # Check if p divides numerator
            if bernoulli.numerator.to_int() % p == 0:
                irregular_pairs.append((idx, p))
        
        return irregular_pairs


# Example usage
def demo_bernoulli_and_irregular_primes():
    """Demonstrate computing Bernoulli numbers and classifying primes."""
    bernoulli = BernoulliNumbers()
    
    print("First few Bernoulli numbers in YANS representation:")
    for i in range(10):
        b = bernoulli.get(i)
        print(f"B_{i} = {b}")
    
    print("\nClassifying some primes:")
    primes_to_check = [3, 5, 7, 11, 13, 17, 37, 59, 67, 101, 103, 131, 149, 157]
    
    regular = []
    irregular = []
    
    for p in primes_to_check:
        is_regular = bernoulli.classify_prime(p)
        if is_regular:
            regular.append(p)
        else:
            irregular.append(p)
            pairs = bernoulli.analyze_irregularity(p)
            print(f"  Prime {p} is irregular: It divides the numerator of B_{pairs[0][0]}")
    
    print(f"\nRegular primes: {regular}")
    print(f"Irregular primes: {irregular}")
    
    # For irregular primes, show the detailed YANS representation
    print("\nDetailed YANS representation for some irregular cases:")
    for p in irregular[:3]:  # Show first few
        pairs = bernoulli.analyze_irregularity(p)
        for idx, prime in pairs:
            b = bernoulli.get(idx)
            print(f"B_{idx} = {b}")
            print(f"Numerator as YANS: {b.numerator}")
            print(f"Denominator as YANS: {b.denominator}")
            print(f"Prime {p} divides the numerator: {b.numerator.to_int() % p == 0}")
            print()


if __name__ == "__main__":
    demo_bernoulli_and_irregular_primes()
