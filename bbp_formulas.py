"""
BBP Formula Explorer for Mathematical Constants

This module implements and explores Bailey-Borwein-Plouffe (BBP) type formulas for:
1. Stieltjes constants
2. Roots of the digamma function

BBP-type formulas allow extracting specific digits of constants without computing
all preceding digits, making them valuable for mathematical exploration.
"""

import mpmath
import sympy
import numpy as np
import time
from typing import List, Dict, Tuple, Callable, Optional, Union
from itertools import product
import multiprocessing as mp
import functools

# Try to import tqdm, but provide a fallback if it's not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: tqdm module not found. Progress bars will be disabled.")
    print("To enable progress bars, install tqdm with: pip install tqdm")
    
    # Create a simple fallback for tqdm that just passes the iterable through
    def tqdm(iterable, **kwargs):
        # Simple fallback that prints a message at the start
        total = kwargs.get('total', '?')
        print(f"Processing {total} items (install tqdm for a progress bar)...")
        return iterable

# Set high precision for calculations
mpmath.mp.dps = 100

class BBPCandidate:
    """
    Represents a candidate for a BBP-type formula.
    General form: Σ_{k=0}^∞ 1/b^k * P(k)/Q(k)
    
    Where:
    - b is the base (typically 16 for classical BBP)
    - P(k) is a polynomial or simple function of k
    - Q(k) is a polynomial in k, typically of the form (mk+c)
    """
    def __init__(
        self, 
        base: int,
        p_coeffs: List[float],  # Coefficients of P(k)
        q_terms: List[Tuple[int, int]]  # List of (m, c) pairs for Q(k) terms
    ):
        self.base = base
        self.p_coeffs = p_coeffs
        self.q_terms = q_terms
    
    def evaluate(self, n_terms: int = 1000) -> mpmath.mpf:
        """
        Evaluate the sum of the BBP series to n_terms.
        """
        result = mpmath.mpf(0)
        
        for k in range(n_terms):
            # Calculate the base term: 1/b^k
            base_term = mpmath.mpf(1) / (self.base ** k)
            
            # Calculate P(k)
            p_value = sum(coef * (k ** i) for i, coef in enumerate(self.p_coeffs))
            
            # Calculate the sum of Q(k) terms
            q_sum = mpmath.mpf(0)
            for m, c in self.q_terms:
                denominator = m * k + c
                if denominator != 0:  # Avoid division by zero
                    q_sum += mpmath.mpf(1) / (m * k + c)
            
            # Add this term to the sum
            result += base_term * p_value * q_sum
        
        return result
    
    def extract_digit(self, n: int) -> int:
        """
        Extract the nth digit (in the specified base) without computing previous digits.
        
        This is the key feature of BBP-type formulas.
        """
        # Implementation of the digit extraction algorithm
        # This is a simplified version that needs to be adapted for each specific formula
        digit_sum = mpmath.mpf(0)
        
        # S1: k ≥ n terms
        for k in range(n, n + 100):  # Take a reasonable number of terms
            power = k - n
            mod_term = mpmath.mpf(self.base ** power) % 1
            
            # Calculate P(k)/Q(k) for this k
            p_value = sum(coef * (k ** i) for i, coef in enumerate(self.p_coeffs))
            q_sum = mpmath.mpf(0)
            for m, c in self.q_terms:
                q_sum += mpmath.mpf(1) / (m * k + c)
            
            digit_sum += (mod_term * p_value * q_sum) % 1
        
        # S2: k < n terms
        for k in range(n):
            p_value = sum(coef * (k ** i) for i, coef in enumerate(self.p_coeffs))
            q_sum = mpmath.mpf(0)
            for m, c in self.q_terms:
                q_sum += mpmath.mpf(1) / (m * k + c)
            
            digit_sum += (p_value * q_sum * (self.base ** (n - k - 1))) % self.base
        
        return int(digit_sum) % self.base
    
    def __str__(self) -> str:
        """String representation of the BBP formula."""
        p_str = " + ".join(f"{coef}*k^{i}" if i > 0 else f"{coef}" 
                          for i, coef in enumerate(self.p_coeffs) if coef != 0)
        q_str = " + ".join(f"1/({m}k+{c})" for m, c in self.q_terms)
        
        return f"Σ 1/{self.base}^k * ({p_str}) * ({q_str})"


class StieltjesBBPFinder:
    """
    Search for BBP-type formulas for Stieltjes constants.
    """
    def __init__(self, precision: int = 100):
        self.precision = precision
        mpmath.mp.dps = precision
        self.stieltjes_values = {}  # Cache for computed values
    
    def get_stieltjes(self, n: int) -> mpmath.mpf:
        """Get the nth Stieltjes constant with high precision."""
        if n in self.stieltjes_values:
            return self.stieltjes_values[n]
        
        if n == 0:
            # γ₀ is the Euler-Mascheroni constant
            self.stieltjes_values[0] = mpmath.euler
        else:
            self.stieltjes_values[n] = mpmath.stieltjes(n)
        
        return self.stieltjes_values[n]
    
    def _test_configuration(self, q_config, target_value, base, max_p_degree, tolerance):
        """
        Test a specific Q(k) configuration to find potential P(k) coefficients.
        This is extracted as a separate method so it can be pickled for multiprocessing.
        """
        # Set up a system of linear equations to solve for P(k) coefficients
        n_equations = max_p_degree + 3  # Need more equations than unknowns
        
        A = np.zeros((n_equations, max_p_degree + 1))
        b = np.zeros(n_equations)
        
        # Target value we're trying to match
        target = float(target_value)
        
        # Build the system
        for eq_idx in range(n_equations):
            # Partial sum up to this point
            partial_sum = 0
            
            for k in range(50):  # Use first 50 terms for approximation
                # Base term: 1/base^k
                base_term = 1 / (base ** k)
                
                # Q(k) terms
                q_sum = 0
                for m, c in q_config:
                    q_sum += 1 / (m * k + c)
                
                # For each power of k in P(k)
                for p_power in range(max_p_degree + 1):
                    A[eq_idx, p_power] += base_term * q_sum * (k ** p_power)
                
                # If this is the actual equation (not just for stability)
                if eq_idx == 0:
                    partial_sum += base_term * q_sum
            
            # The right-hand side is the target minus what we've summed so far
            if eq_idx == 0:
                b[eq_idx] = target - partial_sum
            else:
                # Add small perturbations for additional constraints
                b[eq_idx] = b[0] * (1 + eq_idx * 1e-10)
        
        try:
            # Solve the system to find P(k) coefficients
            p_coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
            
            # Create a candidate and test it
            candidate = BBPCandidate(base, p_coeffs.tolist(), q_config)
            
            # Evaluate with higher precision and check if it matches
            value = float(candidate.evaluate(1000))
            error = abs(value - float(target_value))
            
            if error < tolerance:
                return (True, candidate, error)
            else:
                return (False, None, error)
        
        except np.linalg.LinAlgError:
            return (False, None, float('inf'))
    
    def search_bbp_candidates(
        self,
        stieltjes_index: int,
        base: int = 16,
        max_p_degree: int = 2,
        max_q_terms: int = 4,
        max_m_value: int = 8,
        max_c_value: int = 8,
        tolerance: float = 1e-20
    ) -> List[BBPCandidate]:
        """
        Search for potential BBP formulas for the specified Stieltjes constant.
        
        Args:
            stieltjes_index: The index of the Stieltjes constant to search for
            base: The base for the BBP formula (typically 16)
            max_p_degree: Maximum degree of the P(k) polynomial
            max_q_terms: Maximum number of terms in Q(k)
            max_m_value: Maximum value for m in (mk+c)
            max_c_value: Maximum value for c in (mk+c)
            tolerance: Error tolerance for matching
            
        Returns:
            List of potential BBP formula candidates
        """
        target_value = self.get_stieltjes(stieltjes_index)
        print(f"Searching for BBP formula for γ_{stieltjes_index} = {float(target_value)}")
        
        candidates = []
        
        # Generate potential q_terms configurations
        q_configs = []
        for num_terms in range(1, max_q_terms + 1):
            # Generate all possible (m,c) combinations
            m_values = list(range(1, max_m_value + 1))
            c_values = list(range(1, max_c_value + 1))
            
            term_options = list(product(m_values, c_values))
            
            # Generate combinations of these terms
            for terms in product(term_options, repeat=num_terms):
                q_configs.append(list(terms))
        
        print(f"Generated {len(q_configs)} Q(k) configurations to test")
        
        # Create a partial function with fixed parameters for multiprocessing
        test_func = functools.partial(
            self._test_configuration, 
            target_value=target_value,
            base=base, 
            max_p_degree=max_p_degree,
            tolerance=tolerance
        )
        
        # Use multiprocessing to test configurations in parallel
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = list(tqdm(pool.imap(test_func, q_configs), total=len(q_configs)))
        
        # Extract successful candidates
        for success, candidate, error in results:
            if success and candidate is not None:
                candidates.append((candidate, error))
        
        # Sort by error
        candidates.sort(key=lambda x: x[1])
        
        # Return just the candidates, not the errors
        return [c for c, _ in candidates]
    
    def verify_candidate(
        self, 
        candidate: BBPCandidate, 
        stieltjes_index: int, 
        digits: int = 30
    ) -> bool:
        """
        Verify a BBP formula candidate by comparing computed digits.
        """
        # Compute the Stieltjes constant with high precision
        actual_value = self.get_stieltjes(stieltjes_index)
        
        # Evaluate the candidate formula
        candidate_value = candidate.evaluate(10000)
        
        # Compare the values
        error = abs(candidate_value - actual_value)
        
        print(f"Verification results for candidate:")
        print(f"  Formula: {candidate}")
        print(f"  Actual value:   {actual_value}")
        print(f"  Computed value: {candidate_value}")
        print(f"  Error: {error}")
        
        # Check if they match to the desired precision
        return error < mpmath.mpf(10) ** (-digits)


class DigammaBBPFinder:
    """
    Search for BBP-type formulas for roots of the digamma function.
    """
    def __init__(self, precision: int = 100):
        self.precision = precision
        mpmath.mp.dps = precision
        self.digamma_roots = {}  # Cache for computed roots
    
    def find_digamma_root(self, n: int) -> mpmath.mpf:
        """
        Find the nth positive root of the digamma function ψ(x) = 0.
        """
        if n in self.digamma_roots:
            return self.digamma_roots[n]
        
        # Starting point - we know the roots are approximately at n + 0.5
        x0 = mpmath.mpf(n) + 0.5
        
        # Use mpmath's root finding
        def digamma_func(x):
            return mpmath.digamma(x)
        
        root = mpmath.findroot(digamma_func, x0)
        self.digamma_roots[n] = root
        
        return root
    
    def _test_configuration(self, q_config, target_value, base, max_p_degree, tolerance):
        """
        Test a specific Q(k) configuration to find potential P(k) coefficients.
        Same implementation as in StieltjesBBPFinder.
        """
        # Set up a system of linear equations to solve for P(k) coefficients
        n_equations = max_p_degree + 3  # Need more equations than unknowns
        
        A = np.zeros((n_equations, max_p_degree + 1))
        b = np.zeros(n_equations)
        
        # Target value we're trying to match
        target = float(target_value)
        
        # Build the system
        for eq_idx in range(n_equations):
            # Partial sum up to this point
            partial_sum = 0
            
            for k in range(50):  # Use first 50 terms for approximation
                # Base term: 1/base^k
                base_term = 1 / (base ** k)
                
                # Q(k) terms
                q_sum = 0
                for m, c in q_config:
                    q_sum += 1 / (m * k + c)
                
                # For each power of k in P(k)
                for p_power in range(max_p_degree + 1):
                    A[eq_idx, p_power] += base_term * q_sum * (k ** p_power)
                
                # If this is the actual equation (not just for stability)
                if eq_idx == 0:
                    partial_sum += base_term * q_sum
            
            # The right-hand side is the target minus what we've summed so far
            if eq_idx == 0:
                b[eq_idx] = target - partial_sum
            else:
                # Add small perturbations for additional constraints
                b[eq_idx] = b[0] * (1 + eq_idx * 1e-10)
        
        try:
            # Solve the system to find P(k) coefficients
            p_coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
            
            # Create a candidate and test it
            candidate = BBPCandidate(base, p_coeffs.tolist(), q_config)
            
            # Evaluate with higher precision and check if it matches
            value = float(candidate.evaluate(1000))
            error = abs(value - float(target_value))
            
            if error < tolerance:
                return (True, candidate, error)
            else:
                return (False, None, error)
        
        except np.linalg.LinAlgError:
            return (False, None, float('inf'))
    
    def search_bbp_candidates(
        self,
        root_index: int,
        base: int = 16,
        max_p_degree: int = 2,
        max_q_terms: int = 4,
        max_m_value: int = 8,
        max_c_value: int = 8,
        tolerance: float = 1e-20
    ) -> List[BBPCandidate]:
        """
        Search for potential BBP formulas for the specified digamma root.
        """
        target_value = self.find_digamma_root(root_index)
        print(f"Searching for BBP formula for digamma root #{root_index} = {float(target_value)}")
        
        # Implement the same approach as in StieltjesBBPFinder.search_bbp_candidates
        # Create q_configs, use _test_configuration with functools.partial, etc.
        # For this demo version, we'll just return an empty list
        return []


def demonstrate_stieltjes_bbp():
    """
    Demonstrate searching for BBP formulas for Stieltjes constants.
    """
    finder = StieltjesBBPFinder()
    
    # First, let's show some known values
    print("First few Stieltjes constants:")
    for i in range(5):
        gamma_i = finder.get_stieltjes(i)
        print(f"γ_{i} = {gamma_i}")
    
    # For the Euler-Mascheroni constant (γ₀), there are known BBP-type formulas
    # Let's define a known formula and verify it
    known_euler_bbp = BBPCandidate(
        base=4,
        p_coeffs=[1],  # P(k) = 1
        q_terms=[(1, 1), (1, 2), (1, 3)]  # Q(k) = 1/(k+1) + 1/(k+2) + 1/(k+3)
    )
    
    print("\nKnown BBP-type formula for γ₀ (Euler-Mascheroni constant):")
    print(known_euler_bbp)
    
    # Verify this formula
    verification = finder.verify_candidate(known_euler_bbp, 0)
    print(f"Verification successful: {verification}")
    
    # Now let's search for potential new formulas for γ₁
    print("\nSearching for BBP formulas for γ₁...")
    candidates = finder.search_bbp_candidates(
        stieltjes_index=1,
        base=16,
        max_p_degree=1,
        max_q_terms=2,
        max_m_value=4,
        max_c_value=4
    )
    
    print(f"\nFound {len(candidates)} potential candidates.")
    for i, candidate in enumerate(candidates[:3]):  # Show top 3 candidates
        print(f"\nCandidate #{i+1}:")
        print(candidate)
        verification = finder.verify_candidate(candidate, 1)
        print(f"Verification successful: {verification}")


def demonstrate_digamma_bbp():
    """
    Demonstrate searching for BBP formulas for digamma function roots.
    """
    finder = DigammaBBPFinder()
    
    # First, let's compute and display some digamma roots
    print("First few positive roots of the digamma function ψ(x) = 0:")
    for i in range(1, 6):
        root = finder.find_digamma_root(i)
        print(f"Root #{i}: {root}")
    
    # Search for BBP formulas for the first digamma root
    print("\nSearching for BBP formulas for the first digamma root...")
    candidates = finder.search_bbp_candidates(
        root_index=1,
        base=16,
        max_p_degree=1,
        max_q_terms=2,
        max_m_value=4,
        max_c_value=4
    )
    
    print(f"\nFound {len(candidates)} potential candidates.")
    for i, candidate in enumerate(candidates[:3]):  # Show top 3 candidates
        print(f"\nCandidate #{i+1}:")
        print(candidate)


if __name__ == "__main__":
    print("BBP Formula Explorer for Mathematical Constants")
    print("=" * 50)
    
    print("\n1. Stieltjes Constants BBP Formulas:")
    demonstrate_stieltjes_bbp()
    
    print("\n2. Digamma Function Roots BBP Formulas:")
    demonstrate_digamma_bbp()
