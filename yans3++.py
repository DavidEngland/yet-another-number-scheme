import sympy
import math
import cmath

class YANSNumber:
    """
    Pure integer YANS representation.
    The first exponent is for -1, followed by exponents for 2, 3, 5, ...
    """
    def __init__(self, exponents):
        self.exponents = exponents  # list of ints

    def __str__(self):
        exp_str = "|".join(str(e) for e in self.exponents)
        return f"[{exp_str}]"

    def __eq__(self, other):
        return isinstance(other, YANSNumber) and self.exponents == other.exponents

    def __hash__(self):
        return hash(tuple(self.exponents))

    def __pow__(self, exponent):
        if exponent == 0:
            return YANSNumber([0])
        new_exponents = [e * exponent for e in self.exponents]
        return YANSNumber(new_exponents)

    def __mul__(self, other):
        max_len = max(len(self.exponents), len(other.exponents))
        a = self.exponents + [0] * (max_len - len(self.exponents))
        b = other.exponents + [0] * (max_len - len(other.exponents))
        new_exponents = [x + y for x, y in zip(a, b)]
        return YANSNumber(new_exponents)

    def __truediv__(self, other):
        max_len = max(len(self.exponents), len(other.exponents))
        a = self.exponents + [0] * (max_len - len(self.exponents))
        b = other.exponents + [0] * (max_len - len(other.exponents))
        new_exponents = [x - y for x, y in zip(a, b)]
        return YANSNumber(new_exponents)

    def to_int(self):
        if self.exponents == [0]:
            return 1
        primes = [-1] + list(sympy.primerange(2, 2 + len(self.exponents) - 1))
        value = 1
        for p, e in zip(primes, self.exponents):
            value *= p ** e
        return int(value)

    def to_base(self, base=10):
        n = abs(self.to_int())
        if base < 2 or base > 128:
            raise ValueError("Base must be between 2 and 128")
        charset = (
            '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
            '!@#$%^&*()-_=+[]{}|;:,.<>?/`~"\'\\'
        )
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

    def to_factor_string(self):
        """
        Returns a string showing the traditional prime factorization,
        e.g., '-1 * 2^2 * 3^1' for [1|2|1].
        """
        if self.exponents == [0]:
            return "1"
        factors = []
        # -1 is treated as the first "prime"
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

def yans_representation(n):
    """
    Returns the YANS representation of an integer n as a YANSNumber object.
    The first exponent is for -1, followed by exponents for 2, 3, 5, ...
    Example: str(yans_representation(12)) -> '[0|2|1]'
             str(yans_representation(-12)) -> '[1|2|1]'
    """
    if n == 0:
        raise ValueError("n must be nonzero")
    abs_n = abs(n)
    sign_exp = 1 if n < 0 else 0
    if abs_n == 1:
        return YANSNumber([sign_exp])
    primes = list(sympy.primerange(2, math.isqrt(abs_n) + 1))
    exponents = []
    remainder = abs_n
    for p in primes:
        count = 0
        while remainder % p == 0:
            remainder //= p
            count += 1
        exponents.append(count)
        if remainder == 1:
            break
    if remainder > 1:
        exponents.append(1)
    while exponents and exponents[-1] == 0:
        exponents.pop()
    return YANSNumber([sign_exp] + exponents)

def yans_euler(a, b):
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
    def __init__(self, real, imag):
        if not isinstance(real, YANSNumber) or not isinstance(imag, YANSNumber):
            raise TypeError("real and imag must be YANSNumber instances")
        self.real = real
        self.imag = imag

    def __str__(self):
        real_str = str(self.real)
        imag_str = str(self.imag)
        return f"{real_str} + {imag_str}i"

    def to_complex(self):
        return complex(self.real.to_int(), self.imag.to_int())

    def __pow__(self, exponent):
        """
        Raises the complex number to a power using De Moivre's theorem.
        Returns a new YANSComplex with real and imag as floats.
        """
        z = self.to_complex()
        result = z ** exponent
        return (result.real, result.imag)

class YANSQuaternion:
    """
    Quaternions as four YANSNumber objects (w, x, y, z).
    """
    def __init__(self, w, x, y, z):
        if not all(isinstance(a, YANSNumber) for a in (w, x, y, z)):
            raise TypeError("All components must be YANSNumber instances")
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"{self.w} + {self.x}i + {self.y}j + {self.z}k"

    def to_tuple(self):
        return (self.w.to_int(), self.x.to_int(), self.y.to_int(), self.z.to_int())

class YANSClifford:
    """
    Clifford algebra elements as a list of YANSNumber coefficients for basis blades.
    """
    def __init__(self, blades, coeffs):
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

    def to_dict(self):
        return {blade: coeff.to_int() for blade, coeff in zip(self.blades, self.coeffs)}
