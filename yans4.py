import sympy
import math
import cmath
from typing import List, Optional, Tuple

# ============================================================
# Core: YANSNumber
# ============================================================

class YANSNumber:
    """
    Yet Another Number Scheme (YANS): multiplicative integer representation.

    - Exponent vector of primes with slot 0 reserved for -1 (sign).
    - Example: -12 = -1 * 2^2 * 3^1  → exponents = [1, 2, 1]
                 index: -1   2  3
    - Zero is represented as [], one as [0].
    - Addition/subtraction collapse back to integers.
    """

    def __init__(self, exponents: Optional[List[int]] = None):
        self.exponents = exponents if exponents is not None else []

    # Constructors
    @classmethod
    def zero(cls) -> "YANSNumber":
        return cls([])

    @classmethod
    def one(cls) -> "YANSNumber":
        return cls([0])

    # String + repr
    def __str__(self) -> str:
        if not self.exponents:
            return "[0]"  # zero
        return "[" + "|".join(str(e) for e in self.exponents) + "]"

    def __repr__(self) -> str:
        return f"YANSNumber({self.exponents})"

    # Equality + hashing
    def __eq__(self, other: object) -> bool:
        return isinstance(other, YANSNumber) and self.exponents == other.exponents

    def __hash__(self) -> int:
        return hash(tuple(self.exponents))

    # Multiplicative ops
    def __pow__(self, exponent: int) -> "YANSNumber":
        if not self.exponents:  # zero
            return YANSNumber.zero()
        if exponent == 0:
            return YANSNumber.one()
        return YANSNumber([e * exponent for e in self.exponents])

    def __mul__(self, other: "YANSNumber") -> "YANSNumber":
        max_len = max(len(self.exponents), len(other.exponents))
        a = self.exponents + [0] * (max_len - len(self.exponents))
        b = other.exponents + [0] * (max_len - len(other.exponents))
        return YANSNumber([x + y for x, y in zip(a, b)])

    def __truediv__(self, other: "YANSNumber") -> "YANSNumber":
        max_len = max(len(self.exponents), len(other.exponents))
        a = self.exponents + [0] * (max_len - len(self.exponents))
        b = other.exponents + [0] * (max_len - len(other.exponents))
        return YANSNumber([x - y for x, y in zip(a, b)])

    # Conversion
    def to_int(self) -> int:
        if not self.exponents:
            return 0
        if self.exponents == [0]:
            return 1
        primes = [-1] + [sympy.prime(i) for i in range(1, len(self.exponents))]
        value = 1
        for p, e in zip(primes, self.exponents):
            value *= p ** e
        return int(value)

    def to_factor_string(self) -> str:
        if not self.exponents:
            return "0"
        if self.exponents == [0]:
            return "1"
        primes = [-1] + [sympy.prime(i) for i in range(1, len(self.exponents))]
        factors = []
        for p, e in zip(primes, self.exponents):
            if e == 0:
                continue
            if p == -1:
                if e % 2 == 1:
                    factors.append("-1")
            else:
                factors.append(f"{p}" if e == 1 else f"{p}^{e}")
        return " * ".join(factors) if factors else "1"

    # Explicit integer-domain addition/subtraction
    def add(self, other: "YANSNumber") -> "YANSNumber":
        return yans_representation(self.to_int() + other.to_int())

    def sub(self, other: "YANSNumber") -> "YANSNumber":
        return yans_representation(self.to_int() - other.to_int())


def yans_representation(n: int) -> YANSNumber:
    """Factorize integer into YANSNumber."""
    if n == 0:
        return YANSNumber.zero()
    abs_n = abs(n)
    sign_exp = 1 if n < 0 else 0
    if abs_n == 1:
        return YANSNumber([sign_exp])

    exponents = []
    remainder = abs_n
    for p in sympy.primerange(2, abs_n + 1):
        count = 0
        while remainder % p == 0:
            remainder //= p
            count += 1
        exponents.append(count)
        if remainder == 1:
            break

    if remainder > 1:  # leftover prime
        prime_list = list(sympy.primerange(2, remainder + 1))
        idx = len(prime_list) - 1
        while len(exponents) < idx:
            exponents.append(0)
        exponents.append(1)

    while exponents and exponents[-1] == 0:
        exponents.pop()

    return YANSNumber([sign_exp] + exponents)


# ============================================================
# YANSComplex
# ============================================================

class YANSComplex:
    """
    Complex number with YANS real + imaginary parts.
    Multiplication/division stay multiplicative.
    Addition/subtraction collapse back to integer YANS representations.
    """

    def __init__(self, real: YANSNumber, imag: YANSNumber):
        if not isinstance(real, YANSNumber) or not isinstance(imag, YANSNumber):
            raise TypeError("real and imag must be YANSNumber")
        self.real = real
        self.imag = imag

    def __str__(self) -> str:
        return f"{self.real} + {self.imag}i"

    def to_complex(self) -> complex:
        return complex(self.real.to_int(), self.imag.to_int())

    # Add/sub collapse to integers
    def add(self, other: "YANSComplex") -> "YANSComplex":
        return YANSComplex(self.real.add(other.real), self.imag.add(other.imag))

    def sub(self, other: "YANSComplex") -> "YANSComplex":
        return YANSComplex(self.real.sub(other.real), self.imag.sub(other.imag))

    # Multiplication stays multiplicative
    def __mul__(self, other: "YANSComplex") -> "YANSComplex":
        a, b = self.real, self.imag
        c, d = other.real, other.imag
        real_part = a * c - b * d
        imag_part = a * d + b * c
        return YANSComplex(real_part, imag_part)

    def __pow__(self, exponent: int) -> Tuple[float, float]:
        z = self.to_complex()
        result = z ** exponent
        return (result.real, result.imag)


# ============================================================
# YANSQuaternion
# ============================================================

class YANSQuaternion:
    """
    Quaternion with YANS components (w + xi + yj + zk).
    Only stores structure and allows conversion.
    """

    def __init__(self, w: YANSNumber, x: YANSNumber, y: YANSNumber, z: YANSNumber):
        if not all(isinstance(c, YANSNumber) for c in (w, x, y, z)):
            raise TypeError("All components must be YANSNumber")
        self.w, self.x, self.y, self.z = w, x, y, z

    def __str__(self) -> str:
        return f"{self.w} + {self.x}i + {self.y}j + {self.z}k"

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.w.to_int(), self.x.to_int(), self.y.to_int(), self.z.to_int())


# ============================================================
# YANSClifford
# ============================================================

class YANSClifford:
    """
    Clifford algebra element:
    blades: basis labels ["1", "e1", "e2", "e12", ...]
    coeffs: YANSNumber coefficients.
    """

    def __init__(self, blades: List[str], coeffs: List[YANSNumber]):
        if len(blades) != len(coeffs):
            raise ValueError("blades and coeffs must match length")
        if not all(isinstance(c, YANSNumber) for c in coeffs):
            raise TypeError("All coeffs must be YANSNumber")
        self.blades = blades
        self.coeffs = coeffs

    def __str__(self) -> str:
        terms = []
        for blade, coeff in zip(self.blades, self.coeffs):
            if coeff.to_int() != 0:
                terms.append(f"{coeff}{blade if blade != '1' else ''}")
        return " + ".join(terms) if terms else "0"

    def to_dict(self) -> dict:
        return {blade: coeff.to_int() for blade, coeff in zip(self.blades, self.coeffs)}


# ============================================================
# Mathematical Constants
# ============================================================

class YANSConstant:
    """
    Represents a mathematical constant (π, e, etc.) that can be
    combined with YANSNumbers for symbolic computation.
    """
    def __init__(self, name: str, approx_value: float, alias: str = None):
        self.name = name
        self.approx_value = approx_value
        self.alias = alias or name
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"YANSConstant({self.name})"


class YANSSymbolic:
    """
    Represents a symbolic expression combining YANSNumbers and constants.
    
    Structure:
    - base: YANSNumber - The base YANS integer
    - constants: Dict[YANSConstant, Tuple[YANSNumber, int]] - Constants with exponents
      where the tuple is (coefficient, exponent)
    """
    def __init__(self, base: YANSNumber, constants: dict = None):
        self.base = base
        self.constants = constants or {}  # Maps constant to (coefficient, exponent)
    
    def __str__(self) -> str:
        parts = [str(self.base)]
        for const, (coeff, exp) in self.constants.items():
            if exp == 0:
                continue
            coeff_str = "" if coeff.to_int() == 1 else f"{coeff}·"
            exp_str = "" if exp == 1 else f"^{exp}"
            parts.append(f"{coeff_str}{const.alias}{exp_str}")
        return " · ".join(parts) if parts else "0"
    
    def __mul__(self, other: "YANSSymbolic") -> "YANSSymbolic":
        """Multiply two symbolic expressions, combining like terms."""
        result_base = self.base * other.base
        result_constants = dict(self.constants)  # copy
        
        # Add other's constants
        for const, (coeff, exp) in other.constants.items():
            if const in result_constants:
                # Add exponents for same constant
                existing_coeff, existing_exp = result_constants[const]
                result_constants[const] = (existing_coeff * coeff, existing_exp + exp)
            else:
                result_constants[const] = (coeff, exp)
        
        return YANSSymbolic(result_base, result_constants)
    
    def to_float(self, precision: int = 15) -> float:
        """Convert symbolic expression to floating-point approximation."""
        result = self.base.to_int()
        for const, (coeff, exp) in self.constants.items():
            result *= coeff.to_int() * (const.approx_value ** exp)
        return result
    
    @classmethod
    def from_constant(cls, constant: YANSConstant) -> "YANSSymbolic":
        """Create a symbolic expression from a single constant."""
        return cls(YANSNumber.one(), {constant: (YANSNumber.one(), 1)})


# Predefined constants
PI = YANSConstant("π", math.pi, "π")
E = YANSConstant("e", math.e, "e")
GOLDEN_RATIO = YANSConstant("φ", (1 + math.sqrt(5)) / 2, "φ")
EULER_MASCHERONI = YANSConstant("γ", 0.57721566490153286, "γ")
CATALAN = YANSConstant("G", 0.915965594177219, "G")
APERY = YANSConstant("ζ(3)", 1.2020569031595942, "ζ(3)")

# Constant creation helper
def yans_pi() -> YANSSymbolic:
    return YANSSymbolic.from_constant(PI)

def yans_e() -> YANSSymbolic:
    return YANSSymbolic.from_constant(E)

def yans_golden_ratio() -> YANSSymbolic:
    return YANSSymbolic.from_constant(GOLDEN_RATIO)


# ============================================================
# Extra: Euler form + complex exponentiation
# ============================================================

def yans_euler(a: YANSNumber, b: YANSNumber) -> Tuple[float, float]:
    """
    Compute e^(a + bi) = e^a * (cos(b) + i sin(b)).
    Returns (real, imag).
    """
    a_val, b_val = a.to_int(), b.to_int()
    exp_a = math.exp(a_val)
    return (exp_a * math.cos(b_val), exp_a * math.sin(b_val))


def complex_pow(base: YANSComplex, exponent: YANSComplex) -> Tuple[float, float]:
    """
    Complex exponentiation: z^w = e^(w log z).
    Returns (real, imag).
    """
    z = base.to_complex()
    w = exponent.to_complex()
    result = cmath.exp(w * cmath.log(z))
    return (result.real, result.imag)