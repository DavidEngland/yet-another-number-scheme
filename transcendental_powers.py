"""
Extension to the Gelfond-Schneider module for analyzing transcendental powers.

This module specifically addresses when powers of transcendental numbers
are also transcendental, with special attention to π and e.
"""

from typing import Union, Optional
import math
from enum import Enum, auto
from gelfond_schneider import AlgebraicNumber
from yans3 import YANSNumber, yans_representation

class TranscendentalConstant(Enum):
    """Known transcendental constants"""
    PI = auto()
    E = auto()
    EULER_MASCHERONI = auto()  # γ, Euler's constant
    
    def __str__(self):
        return {
            self.PI: "π",
            self.E: "e",
            self.EULER_MASCHERONI: "γ"
        }[self]
    
    def value(self) -> float:
        """Return numerical approximation"""
        return {
            self.PI: math.pi,
            self.E: math.e,
            self.EULER_MASCHERONI: 0.57721566490153286
        }[self]

class TranscendentalNumber:
    """
    Represents a transcendental number, either a known constant or a composite expression.
    """
    def __init__(self, 
                 constant: Optional[TranscendentalConstant] = None,
                 expression: Optional[str] = None,
                 value_func: Optional[callable] = None):
        """
        Initialize a transcendental number.
        
        Args:
            constant: One of the known transcendental constants
            expression: Symbolic expression (for composite transcendentals)
            value_func: Function to compute numerical approximation
        """
        self.constant = constant
        self.expression = expression or str(constant) if constant else None
        self._value_func = value_func
        
    @classmethod
    def pi(cls) -> 'TranscendentalNumber':
        """Create π"""
        return cls(constant=TranscendentalConstant.PI)
    
    @classmethod
    def e(cls) -> 'TranscendentalNumber':
        """Create e"""
        return cls(constant=TranscendentalConstant.E)
    
    def value(self) -> float:
        """Return numerical approximation"""
        if self._value_func:
            return self._value_func()
        if self.constant:
            return self.constant.value()
        return float('nan')
    
    def __str__(self) -> str:
        return self.expression or "Transcendental"
    
    def __pow__(self, exponent: Union[AlgebraicNumber, int, float]) -> 'TranscendentalNumber':
        """
        Raise this transcendental to a power and determine if result is transcendental.
        """
        # Convert numeric exponents to AlgebraicNumber
        if isinstance(exponent, (int, float)):
            if float(exponent).is_integer():
                yans_exp = yans_representation(int(exponent))
                exponent = AlgebraicNumber.from_rational(yans_exp)
            else:
                # Floating point exponent - assume it's approximating an algebraic number
                # This is a simplification; in reality we'd need to know if it's algebraic
                exponent = AlgebraicNumber([-float(exponent), 1], 0)
        
        # Check if this power is provably transcendental
        is_transcendental, reason = self._check_transcendental_power(exponent)
        
        # Create result expression
        if self.constant:
            base_expr = str(self.constant)
        else:
            base_expr = f"({self.expression})"
            
        # Create a new TranscendentalNumber for the result
        if exponent.is_rational and exponent.yans_value:
            exp_str = str(exponent.yans_value.to_int())
        else:
            exp_str = str(exponent)
        
        result_expr = f"{base_expr}^{exp_str}"
        if not is_transcendental:
            result_expr += " (unknown if transcendental)"
            
        # Function to compute value
        base_val = self.value()
        exp_val = exponent.value
        value_func = lambda: base_val ** exp_val
        
        return TranscendentalNumber(expression=result_expr, value_func=value_func)
    
    def _check_transcendental_power(self, exponent: AlgebraicNumber) -> tuple:
        """
        Check if self^exponent is provably transcendental.
        
        Returns:
            (is_transcendental, reason) tuple
        """
        # Zero exponent always gives 1, which is algebraic
        if exponent.is_equal_to(AlgebraicNumber.from_rational(yans_representation(0))):
            return (False, "x^0 = 1 is algebraic")
        
        # Lindemann-Weierstrass theorem: if α is algebraic and non-zero, then e^α is transcendental
        if self.constant == TranscendentalConstant.E and exponent.is_rational:
            if not exponent.is_equal_to(AlgebraicNumber.from_rational(yans_representation(0))):
                return (True, "Lindemann-Weierstrass: e^α is transcendental for algebraic α≠0")
        
        # For π and other transcendentals, any non-zero algebraic power yields a transcendental
        if exponent.is_rational and not exponent.is_equal_to(AlgebraicNumber.from_rational(yans_representation(0))):
            return (True, "If t is transcendental and α is algebraic and non-zero, then t^α is transcendental")
        
        # Unknown case
        return (False, "Cannot determine transcendence with current theorems")

def analyze_transcendental_power(base_name: str, exponent: int) -> str:
    """
    Analyze whether a given power of a transcendental constant is transcendental.
    
    Args:
        base_name: 'pi', 'e', or 'gamma'
        exponent: Integer power
        
    Returns:
        Analysis as a string
    """
    base = None
    if base_name.lower() == 'pi':
        base = TranscendentalNumber.pi()
    elif base_name.lower() == 'e':
        base = TranscendentalNumber.e()
    elif base_name.lower() in ('gamma', 'γ'):
        base = TranscendentalNumber(constant=TranscendentalConstant.EULER_MASCHERONI)
    else:
        return f"Unknown transcendental constant: {base_name}"
    
    exp_algebraic = AlgebraicNumber.from_rational(yans_representation(exponent))
    result = base.__pow__(exp_algebraic)
    
    is_transcendental, reason = base._check_transcendental_power(exp_algebraic)
    value = result.value()
    
    output = [
        f"{result.expression} ≈ {value:.12g}",
        f"Transcendental: {'Yes' if is_transcendental else 'Unknown'}",
        f"Reason: {reason}"
    ]
    
    return "\n".join(output)

def main():
    """Demo of transcendental power analysis"""
    print("Analysis of π^2:")
    print(analyze_transcendental_power('pi', 2))
    print("\nAnalysis of π^3:")
    print(analyze_transcendental_power('pi', 3))
    print("\nAnalysis of e^2:")
    print(analyze_transcendental_power('e', 2))
    print("\nAnalysis of e^π:")
    pi_alg = AlgebraicNumber([-math.pi, 0, 1], 0)  # Not actually correct, just for demo
    e = TranscendentalNumber.e()
    result = e.__pow__(pi_alg)
    print(f"{result.expression} ≈ {result.value():.12g}")

if __name__ == "__main__":
    main()
