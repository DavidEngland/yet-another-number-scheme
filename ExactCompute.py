"""
ExactCompute: A framework for exact-precision number storage and computation
specifically optimized for pattern detection in number sequences.

This combines multiple representation strategies beyond YANS, including:
- Symbolic representation (via SymPy)
- Rational numbers with GCD reduction
- Continued fractions
- Algebraic number fields
- Prime factorization (YANS-style)
- Specialized representations for transcendental constants
"""

import sympy
from sympy import S, sympify, symbols, Rational, Integer
from fractions import Fraction
import mpmath
from typing import Dict, List, Union, Optional, Tuple, Any
import pickle
import json
import math
import hashlib
from enum import Enum, auto
from collections import defaultdict, Counter
import numpy as np
from functools import lru_cache

# Set precision for mpmath
mpmath.mp.dps = 100  # 100 digits of precision

class NumberType(Enum):
    """The type of number being represented"""
    INTEGER = auto()
    RATIONAL = auto()
    ALGEBRAIC = auto()
    TRANSCENDENTAL = auto()
    SYMBOLIC = auto()
    SPECIAL = auto()  # Special values like infinity, NaN

class RepresentationMethod(Enum):
    """Different ways to represent a number exactly"""
    SYMPY = auto()           # Pure symbolic representation
    FRACTION = auto()        # Numerator/denominator
    FACTORIZATION = auto()   # Prime factorization (YANS-style)
    CONTINUED_FRACTION = auto() # Continued fraction
    MINIMAL_POLYNOMIAL = auto() # For algebraic numbers
    EXACT_VALUE = auto()     # Exact value (for simple numbers)
    FORMULA = auto()         # Formula-based representation
    RECURRENCE = auto()      # Defined by a recurrence relation

class ExactNumber:
    """
    A number represented exactly using one or more exact representations.
    Automatically selects the most efficient representation(s).
    """
    def __init__(
        self, 
        value: Any,
        number_type: Optional[NumberType] = None,
        preferred_rep: Optional[RepresentationMethod] = None
    ):
        self.representations = {}
        self.number_type = number_type
        self._numeric_cache = None
        self._hash_cache = None
        
        # Initial representation based on input type
        if isinstance(value, int):
            self._init_from_int(value)
        elif isinstance(value, Fraction) or isinstance(value, sympy.Rational):
            self._init_from_rational(value)
        elif isinstance(value, float):
            self._init_from_float(value)
        elif isinstance(value, str):
            self._init_from_string(value)
        elif isinstance(value, list) and all(isinstance(x, int) for x in value):
            # Assume YANS representation
            self._init_from_factorization(value)
        elif isinstance(value, tuple) and len(value) >= 2:
            # Assume continued fraction
            self._init_from_continued_fraction(value)
        else:
            raise TypeError(f"Cannot create ExactNumber from {type(value)}")
        
        # Use preferred representation if specified
        if preferred_rep is not None:
            self.ensure_representation(preferred_rep)
    
    def _init_from_int(self, value: int) -> None:
        """Initialize from an integer"""
        self.number_type = NumberType.INTEGER
        self.representations[RepresentationMethod.EXACT_VALUE] = value
        self.representations[RepresentationMethod.SYMPY] = Integer(value)
        
        # For efficiency, only compute factorization for reasonably sized integers
        if abs(value) < 10**12:
            factorization = self._compute_factorization(value)
            self.representations[RepresentationMethod.FACTORIZATION] = factorization
    
    def _init_from_rational(self, value: Union[Fraction, sympy.Rational]) -> None:
        """Initialize from a rational number"""
        self.number_type = NumberType.RATIONAL
        
        if isinstance(value, sympy.Rational):
            num, den = value.p, value.q
            self.representations[RepresentationMethod.SYMPY] = value
            self.representations[RepresentationMethod.FRACTION] = Fraction(num, den)
        else:  # Fraction
            num, den = value.numerator, value.denominator
            self.representations[RepresentationMethod.FRACTION] = value
            self.representations[RepresentationMethod.SYMPY] = Rational(num, den)
        
        # For efficiency, only compute factorization for reasonably sized values
        if abs(num) < 10**12 and abs(den) < 10**12:
            num_factors = self._compute_factorization(num)
            den_factors = self._compute_factorization(den)
            self.representations[RepresentationMethod.FACTORIZATION] = (num_factors, den_factors)
        
        # Compute continued fraction for pattern detection
        self.ensure_representation(RepresentationMethod.CONTINUED_FRACTION)
    
    def _init_from_float(self, value: float) -> None:
        """
        Initialize from a float by trying to find an exact representation
        """
        # Check if it might be a simple rational
        self._numeric_cache = value
        
        # Try to find an exact representation as a rational
        if math.isfinite(value):
            # Use sympy's algorithm to find a rational approximation
            rational = sympy.nsimplify(value, rational=True)
            if abs(float(rational) - value) < 1e-12:
                self._init_from_rational(rational)
                return
        
        # If we can't find a simple rational, use symbolic representation
        self.number_type = NumberType.TRANSCENDENTAL  # Provisional type
        self.representations[RepresentationMethod.SYMPY] = sympify(value)
    
    def _init_from_string(self, value: str) -> None:
        """
        Initialize from a string expression, trying to parse it as an exact value
        """
        try:
            # Try to parse as a symbolic expression
            sympy_expr = sympify(value)
            self._init_from_sympy(sympy_expr)
        except (sympy.SympifyError, TypeError):
            raise ValueError(f"Cannot parse '{value}' as an exact number")
    
    def _init_from_sympy(self, value: sympy.Basic) -> None:
        """Initialize from a sympy expression"""
        self.representations[RepresentationMethod.SYMPY] = value
        
        # Determine number type based on sympy expression
        if value.is_Integer:
            self.number_type = NumberType.INTEGER
            self.representations[RepresentationMethod.EXACT_VALUE] = int(value)
        elif value.is_Rational:
            self.number_type = NumberType.RATIONAL
            self.representations[RepresentationMethod.FRACTION] = Fraction(value.p, value.q)
        elif value.is_algebraic:
            self.number_type = NumberType.ALGEBRAIC
        else:
            self.number_type = NumberType.TRANSCENDENTAL
    
    def _init_from_factorization(self, factors: List[int]) -> None:
        """Initialize from a YANS-style factorization"""
        self.representations[RepresentationMethod.FACTORIZATION] = factors
        
        # Reconstruct the value
        if not factors:
            self.number_type = NumberType.INTEGER
            self.representations[RepresentationMethod.EXACT_VALUE] = 0
            self.representations[RepresentationMethod.SYMPY] = Integer(0)
            return
        
        value = 1
        sign = 1 if factors[0] % 2 == 0 else -1
        
        # Convert factors to value
        primes = [-1] + list(sympy.primerange(2, 2 + len(factors) - 1))
        for prime, exp in zip(primes, factors):
            value *= prime ** exp
        
        self.number_type = NumberType.INTEGER
        self.representations[RepresentationMethod.EXACT_VALUE] = value
        self.representations[RepresentationMethod.SYMPY] = Integer(value)
    
    def _init_from_continued_fraction(self, cf_terms: Tuple[int, ...]) -> None:
        """Initialize from a continued fraction representation"""
        self.representations[RepresentationMethod.CONTINUED_FRACTION] = cf_terms
        
        # Convert continued fraction to a rational or symbolic value
        if len(cf_terms) <= 100:  # Only convert if reasonable size
            value = self._cf_to_value(cf_terms)
            
            if isinstance(value, Fraction):
                self._init_from_rational(value)
            else:
                self._init_from_sympy(value)
    
    def _compute_factorization(self, n: int) -> List[int]:
        """Compute the prime factorization in YANS format"""
        if n == 0:
            return []
        
        sign_exp = 1 if n < 0 else 0
        abs_n = abs(n)
        
        if abs_n == 1:
            return [sign_exp]
        
        # Factorize using sympy for efficiency
        factors = sympy.factorint(abs_n)
        max_prime = max(factors.keys())
        
        # Get ordered list of all primes up to max_prime
        prime_list = [-1] + list(sympy.primerange(2, max_prime + 1))
        
        # Build exponent vector
        exponents = [sign_exp]  # First exponent is for -1 (sign)
        for p in prime_list[1:]:  # Skip -1
            exponents.append(factors.get(p, 0))
        
        # Remove trailing zeros
        while exponents and exponents[-1] == 0:
            exponents.pop()
        
        return exponents
    
    def _cf_to_value(self, cf_terms: Tuple[int, ...]) -> Union[Fraction, sympy.Basic]:
        """Convert a continued fraction to a value"""
        if len(cf_terms) == 1:
            return Fraction(cf_terms[0], 1)
        
        # Start with the last term
        result = Fraction(1, cf_terms[-1])
        
        # Work backwards
        for term in reversed(cf_terms[1:-1]):
            result = Fraction(1, term + result)
        
        # Add the integer part
        result = Fraction(cf_terms[0]) + result
        
        return result
    
    def ensure_representation(self, rep_method: RepresentationMethod) -> None:
        """
        Ensure this number has the specified representation, computing it if needed
        """
        if rep_method in self.representations:
            return
        
        # Compute the requested representation
        if rep_method == RepresentationMethod.SYMPY:
            # Try to convert from other representations
            if RepresentationMethod.EXACT_VALUE in self.representations:
                self.representations[rep_method] = Integer(self.representations[RepresentationMethod.EXACT_VALUE])
            elif RepresentationMethod.FRACTION in self.representations:
                frac = self.representations[RepresentationMethod.FRACTION]
                self.representations[rep_method] = Rational(frac.numerator, frac.denominator)
        
        elif rep_method == RepresentationMethod.FRACTION:
            # Try to convert from other representations
            if RepresentationMethod.SYMPY in self.representations:
                sympy_val = self.representations[RepresentationMethod.SYMPY]
                if sympy_val.is_Rational:
                    self.representations[rep_method] = Fraction(sympy_val.p, sympy_val.q)
            elif RepresentationMethod.EXACT_VALUE in self.representations:
                val = self.representations[RepresentationMethod.EXACT_VALUE]
                self.representations[rep_method] = Fraction(val, 1)
        
        elif rep_method == RepresentationMethod.FACTORIZATION:
            # Try to convert from other representations
            if RepresentationMethod.EXACT_VALUE in self.representations:
                val = self.representations[RepresentationMethod.EXACT_VALUE]
                if isinstance(val, int):
                    self.representations[rep_method] = self._compute_factorization(val)
            elif RepresentationMethod.FRACTION in self.representations:
                frac = self.representations[RepresentationMethod.FRACTION]
                num_factors = self._compute_factorization(frac.numerator)
                den_factors = self._compute_factorization(frac.denominator)
                self.representations[rep_method] = (num_factors, den_factors)
        
        elif rep_method == RepresentationMethod.CONTINUED_FRACTION:
            # Generate continued fraction
            if self.number_type in (NumberType.INTEGER, NumberType.RATIONAL):
                if RepresentationMethod.FRACTION in self.representations:
                    frac = self.representations[RepresentationMethod.FRACTION]
                    cf_terms = self._to_continued_fraction(frac.numerator, frac.denominator)
                    self.representations[rep_method] = cf_terms
                elif RepresentationMethod.EXACT_VALUE in self.representations:
                    val = self.representations[RepresentationMethod.EXACT_VALUE]
                    cf_terms = self._to_continued_fraction(val, 1)
                    self.representations[rep_method] = cf_terms
            elif RepresentationMethod.SYMPY in self.representations:
                # For algebraic or transcendental numbers, use mpmath to compute CF
                val = float(self.representations[RepresentationMethod.SYMPY])
                cf_terms = mpmath.nstr(mpmath.continued_fraction(val, 20), n=20)
                self.representations[rep_method] = cf_terms
    
    def _to_continued_fraction(self, numerator: int, denominator: int) -> Tuple[int, ...]:
        """Convert a rational number to its continued fraction representation"""
        if denominator == 0:
            raise ValueError("Denominator cannot be zero")
        
        a, b = abs(numerator), abs(denominator)
        terms = []
        
        # Get the integer part
        terms.append(a // b)
        a %= b
        
        # If there's a remainder, compute the continued fraction
        while a:
            a, b = b, a
            terms.append(a // b)
            a %= b
        
        return tuple(terms)
    
    def to_float(self, precision: int = 53) -> float:
        """
        Convert to a floating-point approximation with specified precision
        """
        if self._numeric_cache is not None:
            return self._numeric_cache
        
        # Try different representations in order of efficiency
        if RepresentationMethod.EXACT_VALUE in self.representations:
            val = self.representations[RepresentationMethod.EXACT_VALUE]
            if isinstance(val, (int, float)):
                self._numeric_cache = float(val)
                return self._numeric_cache
        
        if RepresentationMethod.FRACTION in self.representations:
            frac = self.representations[RepresentationMethod.FRACTION]
            self._numeric_cache = float(frac)
            return self._numeric_cache
        
        if RepresentationMethod.SYMPY in self.representations:
            sympy_val = self.representations[RepresentationMethod.SYMPY]
            self._numeric_cache = float(sympy_val)
            return self._numeric_cache
        
        raise ValueError("Cannot convert to float: no suitable representation")
    
    def __str__(self) -> str:
        """String representation of the number"""
        # Choose the most readable representation
        if RepresentationMethod.SYMPY in self.representations:
            return str(self.representations[RepresentationMethod.SYMPY])
        
        if RepresentationMethod.EXACT_VALUE in self.representations:
            return str(self.representations[RepresentationMethod.EXACT_VALUE])
        
        if RepresentationMethod.FRACTION in self.representations:
            return str(self.representations[RepresentationMethod.FRACTION])
        
        if RepresentationMethod.FACTORIZATION in self.representations:
            factors = self.representations[RepresentationMethod.FACTORIZATION]
            if isinstance(factors, tuple):  # Rational number
                num, den = factors
                return f"[{','.join(map(str, num))}]/[{','.join(map(str, den))}]"
            else:  # Integer
                return f"[{','.join(map(str, factors))}]"
        
        return "ExactNumber(?)"
    
    def __repr__(self) -> str:
        """Detailed representation showing all available representations"""
        parts = [f"ExactNumber({self.number_type})"]
        for rep_type, value in self.representations.items():
            parts.append(f"  {rep_type.name}: {value}")
        return "\n".join(parts)
    
    def __hash__(self) -> int:
        """Hash for dictionary/set usage"""
        if self._hash_cache is not None:
            return self._hash_cache
        
        # Hash based on canonical representation
        if RepresentationMethod.SYMPY in self.representations:
            self._hash_cache = hash(self.representations[RepresentationMethod.SYMPY])
        elif RepresentationMethod.EXACT_VALUE in self.representations:
            self._hash_cache = hash(self.representations[RepresentationMethod.EXACT_VALUE])
        elif RepresentationMethod.FRACTION in self.representations:
            self._hash_cache = hash(self.representations[RepresentationMethod.FRACTION])
        else:
            # Fall back to hashing string representation
            self._hash_cache = hash(str(self))
        
        return self._hash_cache
    
    def __eq__(self, other) -> bool:
        """Equality check"""
        if not isinstance(other, ExactNumber):
            try:
                other = ExactNumber(other)
            except (TypeError, ValueError):
                return False
        
        # Check symbolic representations if available
        if (RepresentationMethod.SYMPY in self.representations and 
            RepresentationMethod.SYMPY in other.representations):
            return self.representations[RepresentationMethod.SYMPY] == other.representations[RepresentationMethod.SYMPY]
        
        # Check exact values if available
        if (RepresentationMethod.EXACT_VALUE in self.representations and 
            RepresentationMethod.EXACT_VALUE in other.representations):
            return self.representations[RepresentationMethod.EXACT_VALUE] == other.representations[RepresentationMethod.EXACT_VALUE]
        
        # Check fractions if available
        if (RepresentationMethod.FRACTION in self.representations and 
            RepresentationMethod.FRACTION in other.representations):
            return self.representations[RepresentationMethod.FRACTION] == other.representations[RepresentationMethod.FRACTION]
        
        # Fall back to numerical comparison
        try:
            return abs(self.to_float() - other.to_float()) < 1e-12
        except (TypeError, ValueError):
            return False


class ExactSequence:
    """
    A sequence of exact numbers with tools for pattern detection
    """
    def __init__(self, values: Optional[List[Any]] = None):
        self.elements = []
        
        if values:
            for val in values:
                if not isinstance(val, ExactNumber):
                    self.elements.append(ExactNumber(val))
                else:
                    self.elements.append(val)
    
    def append(self, value: Any) -> None:
        """Add a value to the sequence"""
        if not isinstance(value, ExactNumber):
            value = ExactNumber(value)
        self.elements.append(value)
    
    def __getitem__(self, idx: int) -> ExactNumber:
        """Get an element by index"""
        return self.elements[idx]
    
    def __len__(self) -> int:
        """Number of elements in the sequence"""
        return len(self.elements)
    
    def detect_patterns(self) -> Dict[str, Any]:
        """
        Analyze the sequence to detect patterns. Returns a dictionary of findings.
        """
        results = {}
        
        # Only analyze if we have enough elements
        if len(self.elements) < 3:
            return {"error": "Need at least 3 elements for pattern detection"}
        
        # Check for arithmetic sequence
        differences = [self.elements[i+1].to_float() - self.elements[i].to_float() 
                       for i in range(len(self.elements)-1)]
        
        if all(abs(diff - differences[0]) < 1e-10 for diff in differences):
            results["arithmetic"] = {
                "detected": True,
                "common_difference": ExactNumber(differences[0])
            }
        
        # Check for geometric sequence
        ratios = [self.elements[i+1].to_float() / self.elements[i].to_float() 
                  for i in range(len(self.elements)-1) 
                  if abs(self.elements[i].to_float()) > 1e-10]
        
        if ratios and all(abs(ratio - ratios[0]) < 1e-10 for ratio in ratios):
            results["geometric"] = {
                "detected": True,
                "common_ratio": ExactNumber(ratios[0])
            }
        
        # Check for recurrence relations
        results["recurrence"] = self._detect_recurrence()
        
        # Check for patterns in continued fractions
        cf_patterns = self._analyze_continued_fractions()
        if cf_patterns:
            results["continued_fraction_patterns"] = cf_patterns
        
        # Check for patterns in prime factorizations
        factorization_patterns = self._analyze_factorizations()
        if factorization_patterns:
            results["factorization_patterns"] = factorization_patterns
        
        return results
    
    def _detect_recurrence(self) -> Dict[str, Any]:
        """
        Try to detect linear recurrence relations
        """
        # Try different orders of recurrence
        max_order = min(5, len(self.elements) // 2)
        
        for order in range(2, max_order + 1):
            # Try to find coefficients
            # This is a simplified version - a real implementation would use
            # more sophisticated methods like the Berlekamp-Massey algorithm
            pass  # Placeholder for actual implementation
        
        return {}  # Placeholder
    
    def _analyze_continued_fractions(self) -> Dict[str, Any]:
        """
        Analyze patterns in the continued fraction representations
        """
        results = {}
        
        # Ensure all elements have continued fraction representations
        for elem in self.elements:
            elem.ensure_representation(RepresentationMethod.CONTINUED_FRACTION)
        
        # Extract continued fractions
        cfs = [elem.representations.get(RepresentationMethod.CONTINUED_FRACTION) 
               for elem in self.elements]
        
        # Check for patterns
        # For example, periodic continued fractions indicate quadratic irrationals
        for i, cf in enumerate(cfs):
            if isinstance(cf, tuple) and len(cf) > 3:
                # Check for periodicity
                for period_len in range(1, len(cf) // 2):
                    pattern = cf[-(period_len):]
                    extended = cf[-(2*period_len):-period_len]
                    if pattern == extended:
                        results[f"element_{i}"] = {
                            "periodic": True,
                            "period_length": period_len,
                            "period": pattern
                        }
        
        return results
    
    def _analyze_factorizations(self) -> Dict[str, Any]:
        """
        Analyze patterns in the prime factorizations
        """
        results = {}
        
        # Ensure all elements have factorization representations
        integers_only = all(elem.number_type == NumberType.INTEGER for elem in self.elements)
        if not integers_only:
            return {}  # Skip if not all integers
        
        for elem in self.elements:
            elem.ensure_representation(RepresentationMethod.FACTORIZATION)
        
        # Extract factorizations
        factorizations = [elem.representations.get(RepresentationMethod.FACTORIZATION) 
                         for elem in self.elements]
        
        # Analyze common factors
        common_primes = set()
        for factorization in factorizations:
            if not factorization:
                continue
            
            # Get the primes that appear
            primes = []
            for i, exp in enumerate(factorization[1:], start=1):  # Skip sign bit
                if exp > 0:
                    # Convert index to prime
                    prime = sympy.prime(i)
                    primes.append(prime)
            
            if not common_primes:
                common_primes = set(primes)
            else:
                common_primes &= set(primes)
        
        if common_primes:
            results["common_prime_factors"] = sorted(common_primes)
        
        # More advanced analysis can be added here
        
        return results
    
    def save(self, filename: str) -> None:
        """
        Save the sequence to a file for later analysis
        """
        # Convert to a serializable format
        data = {
            "elements": [
                {
                    "type": elem.number_type.name,
                    "string_value": str(elem),
                    # Include the most compact representation
                    "representation": self._get_serializable_representation(elem)
                }
                for elem in self.elements
            ]
        }
        
        # Save as JSON
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _get_serializable_representation(self, elem: ExactNumber) -> Dict[str, Any]:
        """Get a serializable representation of an ExactNumber"""
        if RepresentationMethod.EXACT_VALUE in elem.representations:
            return {
                "method": "exact_value",
                "value": elem.representations[RepresentationMethod.EXACT_VALUE]
            }
        
        if RepresentationMethod.FRACTION in elem.representations:
            frac = elem.representations[RepresentationMethod.FRACTION]
            return {
                "method": "fraction",
                "numerator": frac.numerator,
                "denominator": frac.denominator
            }
        
        if RepresentationMethod.FACTORIZATION in elem.representations:
            fact = elem.representations[RepresentationMethod.FACTORIZATION]
            return {
                "method": "factorization",
                "factors": fact if isinstance(fact, list) else fact[0]
            }
        
        # Default to string representation
        return {
            "method": "string",
            "value": str(elem)
        }
    
    @classmethod
    def load(cls, filename: str) -> 'ExactSequence':
        """
        Load a sequence from a file
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        sequence = cls()
        
        for elem_data in data["elements"]:
            rep = elem_data["representation"]
            method = rep["method"]
            
            if method == "exact_value":
                sequence.append(ExactNumber(rep["value"]))
            elif method == "fraction":
                sequence.append(ExactNumber(Fraction(rep["numerator"], rep["denominator"])))
            elif method == "factorization":
                sequence.append(ExactNumber(rep["factors"]))
            else:
                # Fall back to string parsing
                sequence.append(ExactNumber(elem_data["string_value"]))
        
        return sequence


class PatternDetector:
    """
    Tools for detecting mathematical patterns in sequences of numbers
    """
    @staticmethod
    def oeis_lookup(sequence: List[int], max_terms: int = 10) -> List[str]:
        """
        Check if a sequence exists in the OEIS database
        """
        # This would use an API call to OEIS or a local database
        # Placeholder implementation
        return []
    
    @staticmethod
    @lru_cache(maxsize=128)
    def linear_recurrence_finder(sequence: Tuple[float], max_order: int = 5) -> Dict[str, Any]:
        """
        Try to find a linear recurrence relation that generates the sequence
        """
        # Placeholder for more sophisticated implementation
        return {}
    
    @staticmethod
    def polynomial_fit(sequence: List[float]) -> Dict[str, Any]:
        """
        Find a polynomial that fits the sequence
        """
        # Use numpy to fit polynomials of different degrees
        max_degree = min(len(sequence) - 1, 10)
        best_fit = None
        best_degree = 0
        best_error = float('inf')
        
        x = np.arange(len(sequence))
        y = np.array(sequence)
        
        for degree in range(1, max_degree + 1):
            coeffs = np.polyfit(x, y, degree)
            poly = np.poly1d(coeffs)
            error = np.sum((poly(x) - y) ** 2)
            
            # Check if this is a better fit
            if error < best_error and error < 1e-10:
                best_fit = coeffs
                best_degree = degree
                best_error = error
        
        if best_fit is not None:
            return {
                "type": "polynomial",
                "degree": best_degree,
                "coefficients": best_fit.tolist(),
                "error": best_error
            }
        
        return {"type": "no_polynomial_fit"}
    
    @staticmethod
    def find_formula(sequence: ExactSequence, max_terms: int = 10) -> Dict[str, Any]:
        """
        Try to find a formula that generates the sequence
        """
        # This would be a more sophisticated analysis
        # Placeholder implementation
        return {}


class SequenceGenerator:
    """
    Tools for generating mathematical sequences with exact precision
    """
    @staticmethod
    def fibonacci(n: int) -> ExactSequence:
        """Generate Fibonacci sequence"""
        sequence = ExactSequence([1, 1])
        
        for i in range(2, n):
            next_val = sequence[i-1].to_float() + sequence[i-2].to_float()
            sequence.append(ExactNumber(int(next_val)))
        
        return sequence
    
    @staticmethod
    def factorial(n: int) -> ExactSequence:
        """Generate factorial sequence"""
        sequence = ExactSequence([1])
        
        for i in range(1, n):
            next_val = sequence[i-1].to_float() * i
            sequence.append(ExactNumber(int(next_val)))
        
        return sequence
    
    @staticmethod
    def bernoulli(n: int) -> ExactSequence:
        """Generate Bernoulli numbers"""
        sequence = ExactSequence([1, Fraction(1, 2)])
        
        # Use sympy for accurate calculation
        for i in range(2, n):
            if i % 2 == 1:
                sequence.append(ExactNumber(0))
            else:
                b = sympy.bernoulli(i)
                sequence.append(ExactNumber(Fraction(b.p, b.q)))
        
        return sequence
    
    @staticmethod
    def catalan(n: int) -> ExactSequence:
        """Generate Catalan numbers"""
        sequence = ExactSequence([1])
        
        for i in range(1, n):
            next_val = sequence[i-1].to_float() * 2 * (2*i - 1) // (i + 1)
            sequence.append(ExactNumber(int(next_val)))
        
        return sequence
    
    @staticmethod
    def from_recurrence(initial_terms: List[Any], coefficients: List[Any], n: int) -> ExactSequence:
        """
        Generate a sequence from a linear recurrence relation
        a[n] = c[0]*a[n-1] + c[1]*a[n-2] + ... + c[k-1]*a[n-k]
        """
        sequence = ExactSequence(initial_terms)
        
        # Convert coefficients to ExactNumber
        coeff_exact = [ExactNumber(c) for c in coefficients]
        
        k = len(initial_terms)
        
        for i in range(k, n):
            next_term = ExactNumber(0)
            
            for j in range(k):
                term_index = i - k + j
                product = sequence[term_index].to_float() * coeff_exact[j].to_float()
                next_term = ExactNumber(next_term.to_float() + product)
            
            sequence.append(next_term)

        return sequence