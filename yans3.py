import sympy
import math
import cmath
from typing import List, Optional, Tuple

class YANSNumber:
    """
    Pure integer YANS representation.
    The first exponent is for -1, followed by exponents for 2, 3, 5, ...
    """
    def __init__(self, exponents: Optional[List[int]]):
        self.exponents = exponents if exponents is not None else []

    @classmethod
    def zero(cls) -> 'YANSNumber':
        return cls([])

    def __str__(self):
        if not self.exponents:
            return "[0]"
        exp_str = "|".join(str(e) for e in self.exponents)
        return f"[{exp_str}]"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, YANSNumber) and self.exponents == other.exponents

    def __hash__(self) -> int:
        return hash(tuple(self.exponents))

    def __pow__(self, exponent: int) -> 'YANSNumber':
        if not self.exponents or exponent == 0:
            return YANSNumber([0])
        new_exponents = [e * exponent for e in self.exponents]
        return YANSNumber(new_exponents)

    def __mul__(self, other: 'YANSNumber') -> 'YANSNumber':
        max_len = max(len(self.exponents), len(other.exponents))
        a = self.exponents + [0] * (max_len - len(self.exponents))
        b = other.exponents + [0] * (max_len - len(other.exponents))
        new_exponents = [x + y for x, y in zip(a, b)]
        return YANSNumber(new_exponents)

    def __truediv__(self, other: 'YANSNumber') -> 'YANSNumber':
        max_len = max(len(self.exponents), len(other.exponents))
        a = self.exponents + [0] * (max_len - len(self.exponents))
        b = other.exponents + [0] * (max_len - len(other.exponents))
        new_exponents = [x - y for x, y in zip(a, b)]
        return YANSNumber(new_exponents)

    def to_int(self) -> int:
        if not self.exponents:
            return 0
        if self.exponents == [0]:
            return 1
        primes = [-1] + list(sympy.primerange(2, 2 + len(self.exponents) - 1))
        value = 1
        for p, e in zip(primes, self.exponents):
            value *= p ** e
        return int(value)

    def to_base(self, base: int = 10) -> str:
        n = abs(self.to_int())
        if base < 2:
            raise ValueError("Base must be >= 2")
        # Use digits and lowercase/uppercase letters for base <= 62, otherwise use ASCII chars
        charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if base > len(charset):
            charset += ''.join(chr(i) for i in range(128, 128 + base - len(charset)))
        digits = []
        if n == 0:
            digits.append('0')
        else:
            while n > 0:
                digits.append(charset[n % base])
                n //= base
        digits = ''.join(reversed(digits))
        sign = '-' if self.exponents and self.exponents[0] % 2 == 1 else ''
        return sign + digits

    def to_factor_string(self) -> str:
        if not self.exponents:
            return "0"
        if self.exponents == [0]:
            return "1"
        factors = []
        primes = [-1] + list(sympy.primerange(2, 2 + len(self.exponents) - 1))
        for p, e in zip(primes, self.exponents):
            if e == 0:
                continue
            if p == -1:
                if e % 2 == 1:
                    factors.append("-1")
            else:
                factors.append(f"{p}^{e}")
        return " * ".join(factors) if factors else "1"

def yans_representation(n: int) -> YANSNumber:
    """
    Returns the YANS representation of an integer n as a YANSNumber object.
    The first exponent is for -1, followed by exponents for 2, 3, 5, ...
    Handles zero as YANSNumber([]).
    """
    if n == 0:
        return YANSNumber.zero()
    abs_n = abs(n)
    sign_exp = 1 if n < 0 else 0
    if abs_n == 1:
        return YANSNumber([sign_exp])
    exponents = []
    remainder = abs_n
    primes = sympy.primerange(2, abs_n + 1)
    for p in primes:
        count = 0
        while remainder % p == 0:
            remainder //= p
            count += 1
        exponents.append(count)
        if remainder == 1:
            break
    # If remainder > 1, it's a prime > all previous primes
    if remainder > 1:
        # Find index of remainder in prime sequence
        prime_list = [p for p in sympy.primerange(2, remainder + 1)]
        idx = len(prime_list) - 1
        # Pad exponents to idx
        while len(exponents) < idx:
            exponents.append(0)
        exponents.append(1)
    while exponents and exponents[-1] == 0:
        exponents.pop()
    return YANSNumber([sign_exp] + exponents)

def yans_euler(a: YANSNumber, b: YANSNumber) -> Tuple[float, float]:
    """
    Given two YANSNumber objects a, b, returns (real, imag) as floats:
    e^{a + bi} = e^a * (cos(b) + i sin(b))
    """
    a_val = a.to_int()
    b_val = b.to_int()
    exp_a = math.exp(a_val)
    return (exp_a * math.cos(b_val), exp_a * math.sin(b_val))

class YANSComplex:
    """
    Complex numbers as pairs of YANSNumber objects (real, imag).
    """
    def __init__(self, real: YANSNumber, imag: YANSNumber):
        if not isinstance(real, YANSNumber) or not isinstance(imag, YANSNumber):
            raise TypeError("real and imag must be YANSNumber instances")
        self.real = real
        self.imag = imag

    def __str__(self):
        real_str = str(self.real)
        imag_str = str(self.imag)
        return f"{real_str} + {imag_str}i"

    def to_complex(self) -> complex:
        return complex(self.real.to_int(), self.imag.to_int())

    def __add__(self, other: 'YANSComplex') -> 'YANSComplex':
        return YANSComplex(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other: 'YANSComplex') -> 'YANSComplex':
        return YANSComplex(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other: 'YANSComplex') -> 'YANSComplex':
        # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        a, b = self.real, self.imag
        c, d = other.real, other.imag
        real_part = a * c - b * d
        imag_part = a * d + b * c
        return YANSComplex(real_part, imag_part)

    def __pow__(self, exponent: int) -> Tuple[float, float]:
        z = self.to_complex()
        result = z ** exponent
        return (result.real, result.imag)

class YANSQuaternion:
    """
    Quaternions as four YANSNumber objects (w, x, y, z).
    """
    def __init__(self, w: YANSNumber, x: YANSNumber, y: YANSNumber, z: YANSNumber):
        if not all(isinstance(a, YANSNumber) for a in (w, x, y, z)):
            raise TypeError("All components must be YANSNumber instances")
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"{self.w} + {self.x}i + {self.y}j + {self.z}k"

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.w.to_int(), self.x.to_int(), self.y.to_int(), self.z.to_int())

class YANSClifford:
    """
    Clifford algebra elements as a dictionary of YANSNumber coefficients for basis blades.
    """
    def __init__(self, blades: List[str], coeffs: List[YANSNumber]):
        if len(blades) != len(coeffs):
            raise ValueError("blades and coeffs must have same length")
        if not all(isinstance(c, YANSNumber) for c in coeffs):
            raise TypeError("All coefficients must be YANSNumber instances")
        self.blades = blades
        self.coeffs = coeffs

    def __str__(self):
        terms = []
        for blade, coeff in zip(self.blades, self.coeffs):
            if coeff.to_int() != 0:
                terms.append(f"{str(coeff)}{blade if blade != '1' else ''}")
        return " + ".join(terms) if terms else "0"

    def to_dict(self) -> dict:
        return {blade: coeff.to_int() for blade, coeff in zip(self.blades, self.coeffs)}

# Operator overloading for YANSNumber
def __add__(self, other: 'YANSNumber') -> 'YANSNumber':
    max_len = max(len(self.exponents), len(other.exponents))
    a = self.to_int()
    b = other.to_int()
    return yans_representation(a + b)

def __sub__(self, other: 'YANSNumber') -> 'YANSNumber':
    max_len = max(len(self.exponents), len(other.exponents))
    a = self.to_int()
    b = other.to_int()
    return yans_representation(a - b)

YANSNumber.__add__ = __add__
YANSNumber.__sub__ = __sub__

def complex_pow(base: YANSComplex, exponent: YANSComplex) -> Tuple[float, float]:
    """
    Computes complex exponentiation z^w where z and w are complex numbers.
    Returns a tuple (real, imag) representing the result.
    
    Example: i^i = e^(-π/2) ≈ 0.2079
    """
    # Convert to Python complex
    z = base.to_complex()
    w = exponent.to_complex()
    
    # Calculate z^w using the formula:
    # z^w = e^(w * ln(z))
    result = cmath.exp(w * cmath.log(z))
    return (result.real, result.imag)
