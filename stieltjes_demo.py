"""
Demonstration of ExactCompute framework for computing, storing, and analyzing
Stieltjes constants with high precision.

Stieltjes constants appear in the Laurent series expansion of the Riemann zeta function:
ζ(s) = 1/(s-1) + ∑((-1)^n * γ_n)/(n!) * (s-1)^n

γ_0 is the Euler-Mascheroni constant.
"""

import mpmath
import sympy
import math
from typing import List, Dict, Any, Optional

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Note: matplotlib not found. Plotting functionality disabled.")
    print("To enable plotting, install matplotlib: python3 -m pip install matplotlib")

from ExactCompute import (
    ExactNumber, ExactSequence, NumberType, 
    RepresentationMethod, PatternDetector
)

# Set precision for calculations
mpmath.mp.dps = 100  # 100 digits of precision

class StieltjesConstants:
    """
    Compute, store, and analyze Stieltjes constants using ExactCompute.
    """
    def __init__(self, max_n: int = 20, precision: int = 100):
        self.max_n = max_n
        self.precision = precision
        self._constants: List[mpmath.mpf] = []
        self._exact_sequence: Optional[ExactSequence] = None
        self._compute_constants()
    
    def _compute_constants(self) -> None:
        """
        Compute Stieltjes constants using mpmath's zeta function.
        """
        mpmath.mp.dps = self.precision
        
        # γ_0 is the Euler-Mascheroni constant
        self._constants = [mpmath.euler]
        
        # For n > 0, use the stieltjes function from mpmath
        for n in range(1, self.max_n):
            gamma_n = mpmath.stieltjes(n)
            self._constants.append(gamma_n)
    
    def get_constant(self, n: int) -> mpmath.mpf:
        """Get the nth Stieltjes constant."""
        if n < 0:
            raise ValueError("n must be non-negative")
        if n >= len(self._constants):
            if n >= self.max_n:
                self.max_n = n + 10
                self._compute_constants()
            else:
                gamma_n = mpmath.stieltjes(n)
                self._constants.append(gamma_n)
        return self._constants[n]
    
    def to_exact_sequence(self) -> ExactSequence:
        """
        Convert the Stieltjes constants to an ExactSequence for analysis.
        """
        if self._exact_sequence is None:
            self._exact_sequence = ExactSequence()
            
            for n in range(len(self._constants)):
                # Try to find the best representation for each constant
                gamma_n = self._constants[n]
                
                # Convert to a float first (for simplicity)
                gamma_float = float(gamma_n)
                
                # Try to find an exact representation
                exact_gamma = self._find_exact_representation(gamma_n, n)
                
                # Add to the sequence
                self._exact_sequence.append(exact_gamma)
        
        return self._exact_sequence
    
    def _find_exact_representation(self, value: mpmath.mpf, n: int) -> ExactNumber:
        """
        Try to find an exact representation for a Stieltjes constant.
        """
        # For γ_0 (Euler-Mascheroni constant), use a symbolic representation
        if n == 0:
            # Use the high-precision float but mark it as Euler's constant
            gamma0 = ExactNumber(float(value))
            gamma0.number_type = NumberType.TRANSCENDENTAL
            return gamma0
        
        # For other constants, try to find a pattern or use high precision
        try:
            # For even n > 0, try to detect connections to zeta values
            if n % 2 == 0 and n > 0:
                # Stieltjes constants have connections to zeta values and logs
                # This is a simplified approach - in reality the relationship is complex
                exact_rep = ExactNumber(float(value))
                exact_rep.number_type = NumberType.TRANSCENDENTAL
                return exact_rep
            
            # For odd n > 0, directly use high precision float
            return ExactNumber(float(value))
            
        except (ValueError, TypeError):
            # Fall back to float representation
            return ExactNumber(float(value))
    
    def analyze_constants(self) -> Dict[str, Any]:
        """
        Analyze patterns in the Stieltjes constants.
        """
        sequence = self.to_exact_sequence()
        results = {}
        
        # Check for alternating sign pattern
        signs = [1 if float(self.get_constant(n)) > 0 else -1 for n in range(len(self._constants))]
        alternating = all(signs[i] != signs[i+1] for i in range(len(signs)-1))
        
        if alternating:
            results["sign_pattern"] = "alternating"
        
        # Look for growth patterns in absolute values
        abs_values = [abs(float(self.get_constant(n))) for n in range(len(self._constants))]
        
        # Check for asymptotic growth - Stieltjes constants grow roughly like (n!)
        if len(abs_values) > 5:
            ratios = [abs_values[n+1] / abs_values[n] for n in range(1, len(abs_values)-1)]
            avg_ratio = sum(ratios) / len(ratios)
            results["growth"] = {
                "pattern": "factorial-like" if avg_ratio > 1.5 else "slower than factorial",
                "average_ratio": avg_ratio
            }
        
        # Use PatternDetector for deeper analysis
        float_values = [float(val) for val in self._constants]
        polynomial_fit = PatternDetector.polynomial_fit(float_values[:6])  # Try first few terms
        if polynomial_fit["type"] != "no_polynomial_fit":
            results["polynomial_fit"] = polynomial_fit
        
        return results
    
    def plot_constants(self, n_values: Optional[List[int]] = None) -> None:
        """
        Plot the Stieltjes constants.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Plotting skipped - matplotlib not available")
            print("Install with: python3 -m pip install matplotlib")
            return
            
        if n_values is None:
            n_values = list(range(len(self._constants)))
        
        values = [float(self.get_constant(n)) for n in n_values]
        abs_values = [abs(v) for v in values]
        
        plt.figure(figsize=(12, 10))
        
        # Plot the actual values
        plt.subplot(2, 1, 1)
        plt.plot(n_values, values, 'o-', color='blue')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title("Stieltjes Constants γ_n")
        plt.xlabel("n")
        plt.ylabel("γ_n")
        plt.grid(True)
        
        # Plot absolute values on log scale
        plt.subplot(2, 1, 2)
        plt.semilogy(n_values, abs_values, 'o-', color='red')
        plt.title("Absolute Values of Stieltjes Constants |γ_n|")
        plt.xlabel("n")
        plt.ylabel("|γ_n|")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("stieltjes_constants.png")
        plt.show()
    
    def compute_zeta_approximation(self, s: complex, terms: int) -> complex:
        """
        Approximate the zeta function using Stieltjes constants.
        ζ(s) ≈ 1/(s-1) + ∑((-1)^n * γ_n)/(n!) * (s-1)^n
        """
        if terms > self.max_n:
            # Ensure we have enough constants
            self.max_n = terms
            self._compute_constants()
        
        result = 1 / (s - 1)
        
        for n in range(terms):
            gamma_n = self.get_constant(n)
            term = ((-1)**n * gamma_n) * (s - 1)**n / math.factorial(n)
            result += term
        
        return complex(result)


def demo_stieltjes():
    """
    Demonstrate the computation and analysis of Stieltjes constants.
    """
    print("Computing Stieltjes constants...")
    stieltjes = StieltjesConstants(max_n=15)
    
    print("\nFirst few Stieltjes constants:")
    for n in range(5):
        gamma_n = stieltjes.get_constant(n)
        print(f"γ_{n} = {gamma_n}")
    
    print("\nConverting to ExactSequence for analysis...")
    exact_sequence = stieltjes.to_exact_sequence()
    
    print("\nExact representations:")
    for n in range(5):
        exact_gamma = exact_sequence[n]
        print(f"γ_{n} as ExactNumber: {exact_gamma}")
        if RepresentationMethod.CONTINUED_FRACTION in exact_gamma.representations:
            cf = exact_gamma.representations[RepresentationMethod.CONTINUED_FRACTION]
            print(f"  Continued fraction: {cf}")
    
    print("\nAnalyzing patterns in Stieltjes constants...")
    analysis = stieltjes.analyze_constants()
    for key, value in analysis.items():
        print(f"{key}: {value}")
    
    print("\nSaving the sequence for later analysis...")
    exact_sequence.save("stieltjes_constants.json")
    
    print("\nUsing Stieltjes constants to approximate zeta(2)...")
    zeta2_approx = stieltjes.compute_zeta_approximation(2, 10)
    print(f"ζ(2) ≈ {zeta2_approx}")
    print(f"Actual ζ(2) = {mpmath.zeta(2)} (π²/6 = {math.pi**2/6})")
    
    if MATPLOTLIB_AVAILABLE:
        print("\nPlotting Stieltjes constants...")
        stieltjes.plot_constants()
    else:
        print("\nSkipping plots (matplotlib not available)")


if __name__ == "__main__":
    demo_stieltjes()
