"""
YANS Irrationality Module

An extension to the YANS framework that implements continued fractions
and irrationality measure analysis.

This module provides:
1. Continued fraction representation and conversion
2. Irrationality measure estimation
3. Number classification based on continued fraction properties
4. Integration with the YANS number system
"""

import math
import sympy
import mpmath
from typing import List, Tuple, Dict, Any, Union, Optional, Iterator
from fractions import Fraction
import numpy as np
from yans3 import YANSNumber, yans_representation

# Set precision for high-accuracy calculations
mpmath.mp.dps = 100

class ContinuedFraction:
    """Represents a continued fraction [a0; a1, a2, a3, ...]."""
    
    def __init__(self, terms: List[int], is_complete: bool = False):
        """
        Initialize a continued fraction.
        
        Args:
            terms: List of continued fraction terms [a0; a1, a2, ...]
            is_complete: Whether this is a complete representation (True) or truncated (False)
        """
        self.terms = terms
        self.is_complete = is_complete
        self._convergents = None
        
    @classmethod
    def from_float(cls, x: float, max_terms: int = 30, epsilon: float = 1e-15) -> 'ContinuedFraction':
        """
        Convert a float to a continued fraction.
        
        Args:
            x: The float to convert
            max_terms: Maximum number of terms to compute
            epsilon: Precision threshold for termination
            
        Returns:
            A ContinuedFraction object
        """
        if not math.isfinite(x):
            raise ValueError("Cannot convert infinity or NaN to continued fraction")
            
        terms = []
        is_complete = False
        
        for _ in range(max_terms):
            a = int(x)
            terms.append(a)
            frac = x - a
            
            # Check if we've reached a rational approximation
            if abs(frac) < epsilon:
                is_complete = True
                break
                
            # Avoid division by zero
            if abs(frac) < epsilon:
                break
                
            x = 1 / frac
            
        return cls(terms, is_complete)
    
    @classmethod
    def from_mpf(cls, x: mpmath.mpf, max_terms: int = 50) -> 'ContinuedFraction':
        """
        Convert an mpmath.mpf value to a continued fraction with high precision.
        
        Args:
            x: The mpmath value to convert
            max_terms: Maximum number of terms to compute
            
        Returns:
            A ContinuedFraction object
        """
        if not mpmath.isfinite(x):
            raise ValueError("Cannot convert infinity or NaN to continued fraction")
            
        terms = []
        is_complete = False
        
        for _ in range(max_terms):
            a = int(x)
            terms.append(a)
            frac = x - a
            
            # Check if we've reached a rational approximation
            if abs(frac) < mpmath.mpf('1e-50'):
                is_complete = True
                break
                
            # Avoid division by zero
            if abs(frac) < mpmath.mpf('1e-50'):
                break
                
            x = 1 / frac
            
        return cls(terms, is_complete)
    
    @classmethod
    def from_rational(cls, p: int, q: int) -> 'ContinuedFraction':
        """
        Convert a rational number p/q to a continued fraction.
        
        Args:
            p: Numerator
            q: Denominator
            
        Returns:
            A ContinuedFraction object
        """
        if q == 0:
            raise ZeroDivisionError("Denominator cannot be zero")
            
        terms = []
        is_complete = True
        
        while q != 0:
            terms.append(p // q)
            p, q = q, p % q
            
        return cls(terms, is_complete)
    
    @classmethod
    def from_yans(cls, yans: YANSNumber) -> 'ContinuedFraction':
        """
        Convert a YANSNumber to a continued fraction.
        
        Args:
            yans: A YANSNumber object
            
        Returns:
            A ContinuedFraction object
        """
        n = yans.to_int()
        if n == 0:
            return cls([0], True)
            
        return cls.from_rational(n, 1)
    
    def to_float(self) -> float:
        """
        Convert the continued fraction to a float.
        
        Returns:
            Float approximation of the continued fraction
        """
        if not self.terms:
            return 0.0
            
        result = 0.0
        for a in reversed(self.terms):
            result = a + (1 / result if result != 0 else 0)
            
        return float(result)
    
    def to_mpf(self) -> mpmath.mpf:
        """
        Convert to high-precision mpmath.mpf value.
        
        Returns:
            mpmath.mpf value
        """
        if not self.terms:
            return mpmath.mpf('0')
            
        result = mpmath.mpf('0')
        for a in reversed(self.terms):
            result = a + (mpmath.mpf('1') / result if result != 0 else 0)
            
        return result
    
    def to_fraction(self) -> Fraction:
        """
        Convert to a Python Fraction.
        
        Returns:
            Fraction object representing the continued fraction
        """
        if not self.terms:
            return Fraction(0, 1)
            
        # Get the last convergent
        convergents = self.convergents()
        return Fraction(convergents[-1][0], convergents[-1][1])
    
    def to_yans(self) -> YANSNumber:
        """
        Convert to a YANSNumber.
        
        Returns:
            YANSNumber representing the continued fraction
        """
        if not self.terms:
            return yans_representation(0)
            
        frac = self.to_fraction()
        return yans_representation(frac.numerator) / yans_representation(frac.denominator)
    
    def convergents(self) -> List[Tuple[int, int]]:
        """
        Compute all convergents (p_n/q_n) of the continued fraction.
        
        Returns:
            List of (numerator, denominator) pairs
        """
        if self._convergents is not None:
            return self._convergents
            
        if not self.terms:
            return [(0, 1)]
            
        # Initialize p_-1 = 1, p_0 = a_0, q_-1 = 0, q_0 = 1
        p = [1, self.terms[0]]
        q = [0, 1]
        
        # Calculate convergents
        for i in range(1, len(self.terms)):
            p.append(self.terms[i] * p[i] + p[i-1])
            q.append(self.terms[i] * q[i] + q[i-1])
            
        # Prepare the result as (p, q) pairs, starting from (p_0, q_0)
        self._convergents = list(zip(p[1:], q[1:]))
        return self._convergents
    
    def detect_periodicity(self, max_period: int = None) -> Optional[int]:
        """
        Detect if the continued fraction has a periodic pattern.
        
        Args:
            max_period: Maximum period length to check for
            
        Returns:
            Length of the period if periodic, None otherwise
        """
        if self.is_complete:
            return None  # Complete continued fractions are not periodic
            
        if len(self.terms) < 4:
            return None  # Need at least a few terms to detect periodicity
            
        # Skip the first term (integer part)
        cf_tail = self.terms[1:]
        
        # Determine maximum period to check
        if max_period is None:
            max_period = len(cf_tail) // 2
        else:
            max_period = min(max_period, len(cf_tail) // 2)
            
        # Check for periodicity
        for period in range(1, max_period + 1):
            # Check if the last 'period' terms repeat
            repeats = True
            for i in range(period):
                idx1 = len(cf_tail) - period - period + i
                idx2 = len(cf_tail) - period + i
                
                if idx1 < 0 or cf_tail[idx1] != cf_tail[idx2]:
                    repeats = False
                    break
                    
            if repeats:
                return period
                
        return None
    
    def is_quadratic_irrational(self) -> bool:
        """
        Check if this continued fraction represents a quadratic irrational number.
        
        Returns:
            True if this is a quadratic irrational, False otherwise
        """
        return self.detect_periodicity() is not None
    
    def __str__(self) -> str:
        """String representation of the continued fraction."""
        if not self.terms:
            return "[]"
            
        if len(self.terms) == 1:
            return f"[{self.terms[0]}]"
            
        return f"[{self.terms[0]}; {', '.join(str(t) for t in self.terms[1:])}]"
    
    def __repr__(self) -> str:
        """Detailed representation including completeness."""
        return f"ContinuedFraction({self.terms}, is_complete={self.is_complete})"
    
    def __eq__(self, other) -> bool:
        """Check if two continued fractions are equal."""
        if not isinstance(other, ContinuedFraction):
            return False
            
        # If both are complete, compare terms directly
        if self.is_complete and other.is_complete:
            return self.terms == other.terms
            
        # If one is complete and the other isn't, they might still represent the same value
        try:
            return abs(self.to_mpf() - other.to_mpf()) < mpmath.mpf('1e-50')
        except:
            return False
    
    def __len__(self) -> int:
        """Number of terms in the continued fraction."""
        return len(self.terms)
    
    def __getitem__(self, idx) -> int:
        """Get a specific term."""
        return self.terms[idx]


class IrrationalityMeasure:
    """Tools for estimating and analyzing irrationality measures."""
    
    @staticmethod
    def from_continued_fraction(cf: ContinuedFraction, window: int = 5) -> float:
        """
        Estimate irrationality measure from a continued fraction.
        
        Args:
            cf: A ContinuedFraction object
            window: Window size for local analysis
            
        Returns:
            Estimated irrationality measure
        """
        if len(cf) < window + 1:
            return float('nan')  # Not enough terms
            
        # Get the convergents
        convergents = cf.convergents()
        
        # Calculate approximation errors
        errors = []
        x_mpf = cf.to_mpf()
        
        for i in range(window, len(convergents)):
            p, q = convergents[i]
            error = abs(x_mpf - mpmath.mpf(p) / mpmath.mpf(q))
            
            # Use the formula μ ≈ -log(error)/log(q)
            if error > 0:
                measure = -mpmath.log(error) / mpmath.log(q)
                errors.append(float(measure))
                
        return sum(errors) / len(errors) if errors else float('nan')
    
    @staticmethod
    def from_diophantine(x: Union[float, mpmath.mpf], max_q: int = 10000) -> float:
        """
        Estimate irrationality measure using Diophantine approximation.
        
        Args:
            x: Number to analyze
            max_q: Maximum denominator to consider
            
        Returns:
            Estimated irrationality measure
        """
        # Convert to high precision
        if not isinstance(x, mpmath.mpf):
            x = mpmath.mpf(str(x))
            
        approximations = []
        
        # Find good rational approximations
        for q in range(1, max_q + 1):
            p = int(mpmath.nint(x * q))
            error = abs(x - mpmath.mpf(p) / mpmath.mpf(q))
            approximations.append((p, q, error))
            
        # Sort by error (best approximations first)
        approximations.sort(key=lambda approx: approx[2])
        
        # Select best approximations that represent local minima
        best_approx = []
        for i, (p, q, error) in enumerate(approximations[:100]):
            if i == 0 or q > best_approx[-1][1]:
                best_approx.append((p, q, error))
                
        # Calculate μ for each approximation
        measures = []
        for p, q, error in best_approx:
            # From |x - p/q| < 1/q^μ, we get μ ≈ -log(error)/log(q)
            if error > 0:
                measure = -mpmath.log(error) / mpmath.log(q)
                measures.append(float(measure))
                
        return sum(measures) / len(measures) if measures else float('nan')
    
    @staticmethod
    def classify(measure: float) -> str:
        """
        Classify a number based on its irrationality measure.
        
        Args:
            measure: Irrationality measure
            
        Returns:
            Classification as a string
        """
        if math.isnan(measure):
            return "Unknown"
            
        if measure < 1.1:
            return "Rational (μ = 1)"
            
        if 1.9 <= measure <= 2.1:
            return "Typical irrational (μ = 2)"
            
        if 2.1 < measure < 10:
            return "Algebraic irrational (2 < μ < ∞)"
            
        if measure >= 10:
            return "Liouville-like (very high μ)"
            
        return "Unclassified"
    
    @staticmethod
    def is_liouville_number(cf: ContinuedFraction) -> bool:
        """
        Test if a number might be a Liouville number.
        
        Args:
            cf: Continued fraction
            
        Returns:
            True if likely a Liouville number
        """
        # Liouville numbers have arbitrarily large terms in their continued fraction
        return any(term > 10**6 for term in cf.terms)


class NumberAnalyzer:
    """
    Comprehensive analyzer for number properties using continued fractions
    and irrationality measures.
    """
    
    def __init__(self, value: Union[float, int, YANSNumber, str, mpmath.mpf],
                max_cf_terms: int = 50):
        """
        Initialize with a number to analyze.
        
        Args:
            value: Number to analyze
            max_cf_terms: Maximum continued fraction terms to compute
        """
        self.value = self._convert_to_mpf(value)
        self.max_cf_terms = max_cf_terms
        self.cf = None
        self.irrationality_measure = None
        self.classification = None
        self._analysis_complete = False
        
    def _convert_to_mpf(self, value: Union[float, int, YANSNumber, str, mpmath.mpf]) -> mpmath.mpf:
        """Convert various input types to mpmath.mpf."""
        if isinstance(value, YANSNumber):
            return mpmath.mpf(str(value.to_int()))
        elif isinstance(value, (int, float)):
            return mpmath.mpf(str(value))
        elif isinstance(value, str):
            return mpmath.mpf(value)
        elif isinstance(value, mpmath.mpf):
            return value
        elif hasattr(value, '_mpf_') or hasattr(value, '_mpc_'):  # Handle mpmath constants like pi, e
            return mpmath.mpf(value)
        else:
            raise TypeError(f"Cannot convert {type(value)} to mpmath.mpf")
    
    def analyze(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of the number.
        
        Returns:
            Dictionary of analysis results
        """
        if self._analysis_complete:
            return self._get_results()
            
        # 1. Compute continued fraction
        self.cf = ContinuedFraction.from_mpf(self.value, self.max_cf_terms)
        
        # 2. Check for periodicity
        periodicity = self.cf.detect_periodicity()
        is_quadratic = periodicity is not None
        
        # 3. Estimate irrationality measure
        if not is_quadratic:  # No need for quadratic irrationals (μ = 2)
            self.irrationality_measure = IrrationalityMeasure.from_continued_fraction(self.cf)
        else:
            self.irrationality_measure = 2.0  # Exact value for quadratic irrationals
            
        # 4. Classify the number
        self.classification = IrrationalityMeasure.classify(self.irrationality_measure)
        
        # 5. Additional properties
        self._analysis_complete = True
        return self._get_results()
    
    def _get_results(self) -> Dict[str, Any]:
        """Compile analysis results into a dictionary."""
        periodicity = self.cf.detect_periodicity() if self.cf else None
        
        return {
            "value": str(self.value),
            "continued_fraction": str(self.cf) if self.cf else None,
            "cf_terms": self.cf.terms if self.cf else None,
            "is_periodic": periodicity is not None,
            "period_length": periodicity,
            "is_quadratic_irrational": periodicity is not None,
            "irrationality_measure": self.irrationality_measure,
            "classification": self.classification,
            "convergents": [(p, q) for p, q in self.cf.convergents()] if self.cf else None,
            "is_complete": self.cf.is_complete if self.cf else False,
            "is_liouville": IrrationalityMeasure.is_liouville_number(self.cf) if self.cf else False
        }
    
    def to_yans(self) -> YANSNumber:
        """
        Convert to YANSNumber using the best rational approximation.
        
        Returns:
            YANSNumber representation
        """
        if not self.cf:
            self.cf = ContinuedFraction.from_mpf(self.value, self.max_cf_terms)
            
        return self.cf.to_yans()
    
    def best_rational_approximation(self, max_denominator: int = 1000) -> Tuple[int, int]:
        """
        Find the best rational approximation p/q with q <= max_denominator.
        
        Args:
            max_denominator: Maximum allowed denominator
            
        Returns:
            Tuple (p, q) representing the best approximation
        """
        if not self.cf:
            self.cf = ContinuedFraction.from_mpf(self.value, self.max_cf_terms)
            
        convergents = self.cf.convergents()
        
        # Find the best convergent with denominator <= max_denominator
        best_approx = None
        for p, q in convergents:
            if q <= max_denominator:
                best_approx = (p, q)
                
        return best_approx or (int(self.value), 1)


class YANSIrrational:
    """
    Wrapper class for YANS integration with irrational analysis tools.
    This provides the main interface between YANS and the irrational tools.
    """
    
    @staticmethod
    def continued_fraction(yans: YANSNumber) -> ContinuedFraction:
        """
        Convert a YANSNumber to a continued fraction.
        
        Args:
            yans: YANSNumber to convert
            
        Returns:
            ContinuedFraction representation
        """
        return ContinuedFraction.from_yans(yans)
    
    @staticmethod
    def from_continued_fraction(cf: ContinuedFraction) -> YANSNumber:
        """
        Convert a continued fraction to a YANSNumber.
        
        Args:
            cf: ContinuedFraction to convert
            
        Returns:
            YANSNumber representation
        """
        return cf.to_yans()
    
    @staticmethod
    def analyze(yans: YANSNumber) -> Dict[str, Any]:
        """
        Analyze a YANSNumber for irrationality properties.
        
        Args:
            yans: YANSNumber to analyze
            
        Returns:
            Dictionary of analysis results
        """
        analyzer = NumberAnalyzer(yans)
        return analyzer.analyze()
    
    @staticmethod
    def best_approximation(value: Union[float, str], max_denominator: int = 1000) -> YANSNumber:
        """
        Find the best rational approximation for a value as a YANSNumber.
        
        Args:
            value: Value to approximate
            max_denominator: Maximum denominator to consider
            
        Returns:
            YANSNumber representation of the best approximation
        """
        analyzer = NumberAnalyzer(value)
        p, q = analyzer.best_rational_approximation(max_denominator)
        return yans_representation(p) / yans_representation(q)
    
    @staticmethod
    def is_quadratic_irrational(yans: YANSNumber) -> bool:
        """
        Check if a YANSNumber is a quadratic irrational.
        
        Args:
            yans: YANSNumber to check
            
        Returns:
            True if the number is a quadratic irrational
        """
        # This only works for exact rational YANSNumbers
        # For approximate values, use the continued fraction approach
        cf = ContinuedFraction.from_yans(yans)
        return cf.is_quadratic_irrational()


# Constants with their continued fractions
CONTINUED_FRACTIONS = {
    "π": ContinuedFraction([3, 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1], False),
    "e": ContinuedFraction([2, 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8], False),
    "φ": ContinuedFraction([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], False),
    "√2": ContinuedFraction([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], False),
    "√3": ContinuedFraction([1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1], False),
    "γ": ContinuedFraction([0, 1, 1, 2, 1, 2, 1, 4, 3, 13, 5, 1], False),  # Euler-Mascheroni
}

# Example usage and demo function
def demo():
    """Demonstrate the usage of the YANS irrationality tools."""
    print("YANS Irrationality Module Demo\n")
    
    # 1. Analyze some well-known constants
    print("Analyzing well-known constants:")
    constants = {
        "π": mpmath.pi,
        "e": mpmath.e,
        "√2": mpmath.sqrt(2),
        "φ": (1 + mpmath.sqrt(5)) / 2,  # Golden ratio
    }
    
    for name, value in constants.items():
        analyzer = NumberAnalyzer(value)
        results = analyzer.analyze()
        
        print(f"\n{name} analysis:")
        print(f"  Continued fraction: {results['continued_fraction']}")
        print(f"  First few convergents: {results['convergents'][:3]}")
        print(f"  Irrationality measure: {results['irrationality_measure']:.4f}")
        print(f"  Classification: {results['classification']}")
        print(f"  Periodic: {results['is_periodic']}")
        
    # 2. Convert YANSNumbers to continued fractions
    print("\nConverting YANSNumbers to continued fractions:")
    yans_numbers = [
        yans_representation(42),
        yans_representation(355) / yans_representation(113),  # Good π approximation
    ]
    
    for yans in yans_numbers:
        cf = YANSIrrational.continued_fraction(yans)
        print(f"\n{yans} as continued fraction: {cf}")
        print(f"Value: {cf.to_float()}")
        
    # 3. Find best rational approximations
    print("\nFinding best rational approximations:")
    for name, value in constants.items():
        yans_approx = YANSIrrational.best_approximation(value, 1000)
        print(f"{name} ≈ {yans_approx.to_int()}")
        
    # 4. Create a custom irrational number for analysis
    print("\nAnalyzing a custom irrational number:")
    # Liouville's constant: 0.110001000000000000000001...
    liouville = mpmath.mpf('0.1') + sum(mpmath.mpf('0.1')**(n*(n+1)//2) for n in range(1, 10))
    analyzer = NumberAnalyzer(liouville)
    results = analyzer.analyze()
    
    print(f"Liouville's constant analysis:")
    print(f"  Continued fraction: {results['continued_fraction']}")
    print(f"  Irrationality measure: {results['irrationality_measure']:.4f}")
    print(f"  Classification: {results['classification']}")
    print(f"  Is Liouville number: {results['is_liouville']}")

if __name__ == "__main__":
    demo()
