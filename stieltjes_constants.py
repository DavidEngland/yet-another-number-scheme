"""
Stieltjes Constants Module

This module provides tools for computing, analyzing, and exploring Stieltjes constants
using the YANS framework and continued fraction representations.

Stieltjes constants γₙ appear in the Laurent series expansion of the Riemann zeta function:
ζ(s) = 1/(s-1) + ∑((-1)^n * γₙ)/(n!) * (s-1)^n

γ₀ is the Euler-Mascheroni constant.
"""

import mpmath
import sympy
import math
import time
from typing import List, Dict, Tuple, Union, Optional, Any
import numpy as np
from fractions import Fraction

from yans3 import YANSNumber, yans_representation
from yans_irrational import (
    ContinuedFraction, NumberAnalyzer, YANSIrrational, IrrationalityMeasure
)

# Set precision for calculations
mpmath.mp.dps = 100

class StieltjesConstants:
    """
    Class for computing and analyzing Stieltjes constants.
    """
    def __init__(self, max_n: int = 30, precision: int = 100):
        """
        Initialize with specified maximum index and precision.
        
        Args:
            max_n: Maximum index n for which to compute γₙ
            precision: Number of decimal digits precision
        """
        self.max_n = max_n
        self.precision = precision
        mpmath.mp.dps = precision
        
        # Storage for computed constants
        self._constants: List[mpmath.mpf] = []
        self._yans_approx: Dict[int, YANSNumber] = {}
        self._continued_fractions: Dict[int, ContinuedFraction] = {}
        self._patterns: Dict[str, Any] = {}
        
        # Compute the constants up to max_n
        self._compute_constants()
    
    def _compute_constants(self) -> None:
        """
        Compute Stieltjes constants using mpmath's stieltjes function.
        """
        start_time = time.time()
        print(f"Computing Stieltjes constants up to γ_{self.max_n}...")
        
        # γ₀ is the Euler-Mascheroni constant
        self._constants = [mpmath.euler]
        
        # For n > 0, use the stieltjes function
        for n in range(1, self.max_n + 1):
            gamma_n = mpmath.stieltjes(n)
            self._constants.append(gamma_n)
        
        end_time = time.time()
        print(f"Computation completed in {end_time - start_time:.2f} seconds.")
    
    def get_constant(self, n: int) -> mpmath.mpf:
        """
        Get the nth Stieltjes constant.
        
        Args:
            n: Index of the Stieltjes constant to retrieve
            
        Returns:
            The value of γₙ as an mpmath.mpf
        """
        if n < 0:
            raise ValueError("Index must be non-negative")
        
        if n >= len(self._constants):
            if n > self.max_n:
                self.max_n = n
                self._compute_constants()
            else:
                gamma_n = mpmath.stieltjes(n)
                self._constants.append(gamma_n)
        
        return self._constants[n]
    
    def to_yans(self, n: int, max_denominator: int = 10**15) -> YANSNumber:
        """
        Convert γₙ to a rational approximation as a YANSNumber.
        
        Args:
            n: Index of the Stieltjes constant
            max_denominator: Maximum denominator for the rational approximation
            
        Returns:
            YANSNumber approximation of γₙ
        """
        if n in self._yans_approx:
            return self._yans_approx[n]
        
        # Get the constant
        gamma_n = self.get_constant(n)
        
        # Create a rational approximation
        analyzer = NumberAnalyzer(gamma_n)
        p, q = analyzer.best_rational_approximation(max_denominator)
        
        # Convert to YANSNumber
        yans_approx = yans_representation(p) / yans_representation(q)
        self._yans_approx[n] = yans_approx
        
        return yans_approx
    
    def to_continued_fraction(self, n: int, max_terms: int = 50) -> ContinuedFraction:
        """
        Convert γₙ to a continued fraction.
        
        Args:
            n: Index of the Stieltjes constant
            max_terms: Maximum number of terms in the continued fraction
            
        Returns:
            ContinuedFraction representation of γₙ
        """
        if n in self._continued_fractions:
            return self._continued_fractions[n]
        
        # Get the constant
        gamma_n = self.get_constant(n)
        
        # Convert to continued fraction
        cf = ContinuedFraction.from_mpf(gamma_n, max_terms)
        self._continued_fractions[n] = cf
        
        return cf
    
    def analyze_constant(self, n: int) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of the nth Stieltjes constant.
        
        Args:
            n: Index of the Stieltjes constant
            
        Returns:
            Dictionary with analysis results
        """
        gamma_n = self.get_constant(n)
        analyzer = NumberAnalyzer(gamma_n)
        results = analyzer.analyze()
        
        # Add YANS-specific information
        results["yans_approximation"] = str(self.to_yans(n))
        
        return results
    
    def find_patterns(self, max_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Look for patterns in the sequence of Stieltjes constants.
        
        Args:
            max_n: Maximum index to analyze (defaults to self.max_n)
            
        Returns:
            Dictionary with pattern analysis results
        """
        if max_n is None:
            max_n = self.max_n
        
        if max_n > self.max_n:
            self.max_n = max_n
            self._compute_constants()
        
        # Extract values and sign patterns
        values = [float(self.get_constant(n)) for n in range(max_n + 1)]
        abs_values = [abs(v) for v in values]
        signs = [1 if v > 0 else -1 for v in values]
        
        # Check for sign alternation
        sign_alternating = all(signs[i] != signs[i+1] for i in range(len(signs)-1))
        
        # Check for growth rate
        if len(values) > 5:
            # Compute log(|γₙ|) and look for linear growth, which would indicate factorial-like growth
            log_values = [math.log(abs(v)) if abs(v) > 0 else float('-inf') for v in values[2:]]
            indices = list(range(2, len(values)))
            
            # Use numpy for linear regression on log values
            if len(log_values) > 1 and all(not math.isinf(v) for v in log_values):
                try:
                    slope, intercept = np.polyfit(indices, log_values, 1)
                    growth_rate = math.exp(slope)
                except:
                    growth_rate = None
            else:
                growth_rate = None
        else:
            growth_rate = None
        
        # Check continued fraction patterns
        cf_patterns = {}
        for n in range(min(10, max_n + 1)):
            cf = self.to_continued_fraction(n)
            periodic = cf.detect_periodicity()
            
            if periodic:
                cf_patterns[n] = {
                    "periodic": True,
                    "period": periodic,
                    "pattern": cf.terms[1:1+periodic]
                }
            else:
                # Look for unusually large terms which might indicate high irrationality measure
                large_terms = [t for t in cf.terms if t > 1000]
                if large_terms:
                    cf_patterns[n] = {
                        "periodic": False,
                        "large_terms": large_terms
                    }
        
        # Store and return results
        self._patterns = {
            "sign_alternating": sign_alternating,
            "growth_rate": growth_rate,
            "continued_fraction_patterns": cf_patterns,
            "irrationality_measures": self._analyze_irrationality_measures(max_n)
        }
        
        return self._patterns
    
    def _analyze_irrationality_measures(self, max_n: int) -> Dict[int, float]:
        """
        Analyze irrationality measures for a range of Stieltjes constants.
        
        Args:
            max_n: Maximum index to analyze
            
        Returns:
            Dictionary mapping indices to estimated irrationality measures
        """
        measures = {}
        
        for n in range(min(10, max_n + 1)):
            cf = self.to_continued_fraction(n)
            measure = IrrationalityMeasure.from_continued_fraction(cf)
            measures[n] = measure
        
        return measures
    
    def zeta_approximation(self, s: complex, terms: int = 10) -> complex:
        """
        Approximate the Riemann zeta function using Stieltjes constants.
        
        Args:
            s: Complex argument for ζ(s)
            terms: Number of terms to use in the approximation
            
        Returns:
            Approximation of ζ(s)
        """
        if s == 1:
            raise ValueError("ζ(s) has a pole at s=1")
        
        # Ensure we have enough constants
        if terms > self.max_n:
            self.max_n = terms
            self._compute_constants()
        
        # Laurent series for ζ(s) around s=1
        # ζ(s) = 1/(s-1) + ∑((-1)^n * γₙ)/(n!) * (s-1)^n
        result = 1 / (s - 1)
        
        for n in range(terms):
            gamma_n = self.get_constant(n)
            term = ((-1)**n * gamma_n) * (s - 1)**n / math.factorial(n)
            result += term
        
        return complex(result)
    
    def compare_approximations(self, s: complex, max_terms: int = 10) -> Dict[str, Any]:
        """
        Compare zeta approximations with different numbers of Stieltjes constants.
        
        Args:
            s: Complex argument for ζ(s)
            max_terms: Maximum number of terms to use
            
        Returns:
            Dictionary with comparison results
        """
        if s == 1:
            raise ValueError("ζ(s) has a pole at s=1")
        
        # Compute the actual value using mpmath
        actual_zeta = complex(mpmath.zeta(s))
        
        # Compute approximations with increasing numbers of terms
        approx_results = []
        
        for terms in range(1, max_terms + 1):
            approx = self.zeta_approximation(s, terms)
            error = abs(approx - actual_zeta) / abs(actual_zeta)
            
            approx_results.append({
                "terms": terms,
                "approximation": approx,
                "relative_error": float(error)
            })
        
        return {
            "s": s,
            "actual": actual_zeta,
            "approximations": approx_results
        }
    
    def explore_convergence(self, terms_range: range = range(1, 21, 5)) -> Dict[str, Any]:
        """
        Explore convergence of zeta approximations using Stieltjes constants.
        
        Args:
            terms_range: Range of term counts to test
            
        Returns:
            Dictionary with convergence analysis
        """
        test_points = [
            2,                      # A typical real value
            complex(0.5, 14.135),   # Near the first non-trivial zero
            complex(0.5, 100),      # Higher on the critical line
            complex(2, 3),          # A typical complex value
        ]
        
        results = {}
        
        for s in test_points:
            comparison = self.compare_approximations(s, max(terms_range))
            results[str(s)] = comparison
        
        return results


def explore_stieltjes_continued_fractions() -> None:
    """
    Explore the continued fraction representations of Stieltjes constants.
    """
    stieltjes = StieltjesConstants(max_n=10)
    
    print("\nContinued fraction representations of the first few Stieltjes constants:")
    for n in range(5):
        cf = stieltjes.to_continued_fraction(n)
        cf_str = str(cf)
        # Truncate very long representations for display
        if len(cf_str) > 80:
            cf_str = cf_str[:77] + "..."
            
        print(f"γ_{n} = {cf_str}")
        
        # Show a few convergents
        convergents = cf.convergents()[:3]
        print(f"  First few convergents: {convergents}")
        
        # Check irrationality measure
        measure = IrrationalityMeasure.from_continued_fraction(cf)
        print(f"  Estimated irrationality measure: {measure:.4f}")
        print(f"  Classification: {IrrationalityMeasure.classify(measure)}")
        print()


def explore_stieltjes_zeta_connection() -> None:
    """
    Explore the connection between Stieltjes constants and the Riemann zeta function.
    """
    stieltjes = StieltjesConstants(max_n=15)
    
    print("\nUsing Stieltjes constants to approximate the Riemann zeta function:")
    
    # Test some well-known values
    test_points = [
        (2, "π²/6 ≈ 1.6449..."),
        (3, "ζ(3) ≈ 1.2021... (Apéry's constant)"),
        (4, "π⁴/90 ≈ 1.0823..."),
        (complex(0.5, 14.135), "Near first non-trivial zero")
    ]
    
    for s, description in test_points:
        print(f"\nApproximating ζ({s}) ({description}):")
        
        comparison = stieltjes.compare_approximations(s, 10)
        actual = comparison["actual"]
        
        print(f"  Actual value: {actual}")
        
        for approx in comparison["approximations"]:
            terms = approx["terms"]
            value = approx["approximation"]
            error = approx["relative_error"]
            
            print(f"  With {terms} terms: {value} (relative error: {error:.6e})")


def explore_stieltjes_patterns() -> None:
    """
    Explore patterns in the sequence of Stieltjes constants.
    """
    stieltjes = StieltjesConstants(max_n=15)
    
    print("\nExploring patterns in Stieltjes constants:")
    
    # Print the first few constants
    print("First few Stieltjes constants:")
    for n in range(6):
        gamma_n = stieltjes.get_constant(n)
        print(f"γ_{n} = {gamma_n}")
    
    # Find patterns
    patterns = stieltjes.find_patterns()
    
    # Print results
    print("\nSign alternation:", "Yes" if patterns["sign_alternating"] else "No")
    
    if patterns["growth_rate"]:
        print(f"Growth rate: Approximately {patterns['growth_rate']:.4f} per term")
        print(f"This suggests {'factorial-like' if patterns['growth_rate'] > 1.5 else 'sub-factorial'} growth")
    
    # Check for interesting continued fraction patterns
    cf_patterns = patterns["continued_fraction_patterns"]
    if cf_patterns:
        print("\nInteresting continued fraction patterns:")
        for n, pattern in cf_patterns.items():
            if pattern.get("periodic"):
                print(f"γ_{n} has a periodic continued fraction with period {pattern['period']}")
                print(f"  Repeating pattern: {pattern['pattern']}")
            elif pattern.get("large_terms"):
                print(f"γ_{n} has unusually large terms: {pattern['large_terms']}")
    
    # Check irrationality measures
    print("\nIrrationality measures:")
    for n, measure in patterns["irrationality_measures"].items():
        print(f"γ_{n}: μ ≈ {measure:.4f} ({IrrationalityMeasure.classify(measure)})")


def main() -> None:
    """Main demonstration function."""
    print("YANS Stieltjes Constants Explorer")
    print("=" * 40)
    
    # Create a StieltjesConstants object
    stieltjes = StieltjesConstants(max_n=15)
    
    # Explore continued fraction representations
    explore_stieltjes_continued_fractions()
    
    # Explore connection to zeta function
    explore_stieltjes_zeta_connection()
    
    # Explore patterns
    explore_stieltjes_patterns()
    
    print("\nExploration complete!")


if __name__ == "__main__":
    main()
