"""
Advanced exploration of Stieltjes constants using the ExactCompute framework.
This module builds on stieltjes_demo.py to provide deeper analysis and applications.
"""

import mpmath
import sympy
import math
import json
from typing import List, Dict, Any, Tuple
from ExactCompute import ExactNumber, ExactSequence, NumberType, RepresentationMethod
from stieltjes_demo import StieltjesConstants

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Note: Plotting disabled. Install matplotlib for visualizations.")

class StieltjesExplorer:
    """
    Advanced exploration tools for Stieltjes constants.
    """
    def __init__(self, max_n: int = 30, precision: int = 100):
        self.calculator = StieltjesConstants(max_n, precision)
        self.max_n = max_n
        
    def continued_fraction_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in the continued fraction representations of Stieltjes constants.
        """
        results = {}
        sequence = self.calculator.to_exact_sequence()
        
        # Get continued fractions for each constant
        cf_data = []
        for n in range(min(10, len(sequence))):
            exact_gamma = sequence[n]
            exact_gamma.ensure_representation(RepresentationMethod.CONTINUED_FRACTION)
            if RepresentationMethod.CONTINUED_FRACTION in exact_gamma.representations:
                cf = exact_gamma.representations[RepresentationMethod.CONTINUED_FRACTION]
                cf_data.append({
                    "index": n,
                    "continued_fraction": cf,
                    "length": len(cf),
                    "first_term": cf[0] if len(cf) > 0 else None,
                    "second_term": cf[1] if len(cf) > 1 else None
                })
        
        results["continued_fractions"] = cf_data
        
        # Analyze growth in the first term
        if len(cf_data) > 1:
            first_terms = [d["first_term"] for d in cf_data if d["first_term"] is not None]
            if all(t == 0 for t in first_terms):
                # Analyze second terms (typical for small constants starting with 0)
                second_terms = [d["second_term"] for d in cf_data if d["second_term"] is not None]
                if len(second_terms) > 1 and all(isinstance(t, (int, float)) for t in second_terms):
                    growth_ratio = sum(second_terms[i+1]/second_terms[i] for i in range(len(second_terms)-1)) / (len(second_terms)-1)
                    results["second_term_growth_ratio"] = growth_ratio
        
        return results
    
    def zeta_approximation_errors(self, s_values: List[complex], terms: int = 20) -> Dict[complex, float]:
        """
        Calculate approximation errors when using Stieltjes constants to approximate zeta(s).
        """
        errors = {}
        
        for s in s_values:
            if s == 1:  # Pole at s=1
                continue
                
            # Calculate approximation
            approx = self.calculator.compute_zeta_approximation(s, terms)
            
            # Calculate exact value
            exact = complex(mpmath.zeta(s))
            
            # Calculate relative error
            rel_error = abs(approx - exact) / abs(exact)
            errors[s] = float(rel_error)
            
        return errors
    
    def generate_convergence_data(self, s: complex, max_terms: int = 20) -> List[Dict[str, Any]]:
        """
        Generate data showing how adding more Stieltjes constants improves zeta approximation.
        """
        convergence_data = []
        
        exact = complex(mpmath.zeta(s))
        
        for terms in range(1, max_terms + 1):
            approx = self.calculator.compute_zeta_approximation(s, terms)
            rel_error = abs(approx - exact) / abs(exact)
            
            convergence_data.append({
                "terms": terms,
                "approximation": {
                    "real": approx.real,
                    "imag": approx.imag
                },
                "exact": {
                    "real": exact.real,
                    "imag": exact.imag
                },
                "relative_error": float(rel_error)
            })
            
        return convergence_data
    
    def plot_convergence(self, s: complex, max_terms: int = 20) -> None:
        """
        Plot the convergence of zeta approximation as more terms are added.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Plotting requires matplotlib.")
            return
            
        data = self.generate_convergence_data(s, max_terms)
        terms = [d["terms"] for d in data]
        errors = [d["relative_error"] for d in data]
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(terms, errors, 'o-', color='blue')
        plt.title(f"Convergence of ζ({s}) Approximation")
        plt.xlabel("Number of Terms")
        plt.ylabel("Relative Error (log scale)")
        plt.grid(True)
        plt.savefig(f"zeta_convergence_{abs(s)}.png")
        plt.show()
    
    def plot_critical_strip(self, terms: int = 15) -> None:
        """
        Plot the accuracy of zeta approximation in the critical strip.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Plotting requires matplotlib.")
            return
            
        # Create a grid in the critical strip
        real_vals = np.linspace(0, 1, 20)
        imag_vals = np.linspace(5, 15, 20)
        X, Y = np.meshgrid(real_vals, imag_vals)
        Z = np.zeros_like(X)
        
        # Calculate error at each point
        for i in range(len(real_vals)):
            for j in range(len(imag_vals)):
                s = complex(real_vals[i], imag_vals[j])
                approx = self.calculator.compute_zeta_approximation(s, terms)
                exact = complex(mpmath.zeta(s))
                rel_error = abs(approx - exact) / abs(exact)
                Z[j, i] = np.log10(rel_error)  # Log scale for better visualization
                
        # Plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
        
        ax.set_xlabel('Re(s)')
        ax.set_ylabel('Im(s)')
        ax.set_zlabel('Log10(Relative Error)')
        ax.set_title(f'Zeta Approximation Error in Critical Strip (Terms={terms})')
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig("zeta_critical_strip.png")
        plt.show()
    
    def analyze_relations_to_zeta_zeros(self, num_zeros: int = 10) -> Dict[str, Any]:
        """
        Analyze potential relationships between Stieltjes constants and zeta zeros.
        """
        results = {}
        
        # Get first few non-trivial zeros of zeta
        zeros = []
        for i in range(1, num_zeros + 1):
            zero = complex(0.5, mpmath.zetazero(i).imag)
            zeros.append(zero)
            
        # Get Stieltjes constants
        stieltjes_values = [float(self.calculator.get_constant(n)) for n in range(num_zeros)]
        
        # Calculate correlations
        zero_heights = [z.imag for z in zeros]
        
        # Simple ratios
        ratios = []
        for i in range(min(len(zero_heights), len(stieltjes_values))):
            if stieltjes_values[i] != 0:
                ratios.append(zero_heights[i] / abs(stieltjes_values[i]))
                
        results["zero_stieltjes_ratios"] = ratios
        
        # Look for patterns in ratios
        if len(ratios) > 2:
            ratio_diffs = [ratios[i+1] - ratios[i] for i in range(len(ratios)-1)]
            results["ratio_differences"] = ratio_diffs
            
            # Check if differences grow linearly
            if len(ratio_diffs) > 2:
                diffs_of_diffs = [ratio_diffs[i+1] - ratio_diffs[i] for i in range(len(ratio_diffs)-1)]
                avg_second_diff = sum(diffs_of_diffs) / len(diffs_of_diffs)
                results["average_second_difference"] = avg_second_diff
        
        return results

def demo_advanced_stieltjes():
    """
    Demonstrate the advanced Stieltjes constants exploration.
    """
    print("Creating Stieltjes Explorer...")
    explorer = StieltjesExplorer(max_n=20)
    
    print("\nAnalyzing continued fraction patterns...")
    cf_patterns = explorer.continued_fraction_patterns()
    print(f"Found {len(cf_patterns['continued_fractions'])} continued fractions")
    if "second_term_growth_ratio" in cf_patterns:
        print(f"Second term growth ratio: {cf_patterns['second_term_growth_ratio']}")
    
    print("\nCalculating zeta approximation errors...")
    s_values = [2, 3, 4, complex(0.5, 14.1347)]  # Including a point near the 1st zero
    errors = explorer.zeta_approximation_errors(s_values)
    for s, error in errors.items():
        print(f"ζ({s}) approximation relative error: {error:.6e}")
    
    print("\nGenerating convergence data for ζ(2)...")
    convergence = explorer.generate_convergence_data(2, max_terms=15)
    print(f"With 1 term: error = {convergence[0]['relative_error']:.6e}")
    print(f"With 5 terms: error = {convergence[4]['relative_error']:.6e}")
    print(f"With 15 terms: error = {convergence[14]['relative_error']:.6e}")
    
    if MATPLOTLIB_AVAILABLE:
        print("\nPlotting convergence for ζ(2)...")
        explorer.plot_convergence(2, max_terms=15)
        
        print("\nAnalyzing critical strip approximation...")
        explorer.plot_critical_strip(terms=10)
    
    print("\nAnalyzing relations to zeta zeros...")
    zero_relations = explorer.analyze_relations_to_zeta_zeros(num_zeros=8)
    print(f"Zero-Stieltjes ratios: {zero_relations['zero_stieltjes_ratios']}")
    if "average_second_difference" in zero_relations:
        print(f"Average second difference: {zero_relations['average_second_difference']}")
    
    # Save results for further analysis
    print("\nSaving analysis results...")
    with open("stieltjes_analysis.json", "w") as f:
        json.dump({
            "continued_fraction_patterns": cf_patterns,
            "zeta_approximation_errors": {str(s): err for s, err in errors.items()},
            "zeta_convergence": convergence,
            "zero_relations": zero_relations
        }, f, indent=2)
    
    print("Analysis complete. Results saved to stieltjes_analysis.json")

if __name__ == "__main__":
    demo_advanced_stieltjes()
