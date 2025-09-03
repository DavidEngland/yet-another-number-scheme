import sympy
import math

class YANSNumber:
    _primes = [2]  # Store primes in a class attribute

    @staticmethod
    def _get_primes_up_to_n(n):
        while len(YANSNumber._primes) < n:
            YANSNumber._primes.append(sympy.nextprime(YANSNumber._primes[-1]))
        return YANSNumber._primes[:n]

    def __init__(self, sign, exponents):
        self.sign = sign  # 1, -1, or 0 (for zero)
        self.exponents = tuple(exponents)  # Use a tuple for immutability

    def __str__(self):
        if self.sign == 0:
            return "0"
        sign_str = '+' if self.sign == 1 else '-'
        exp_str = "|".join(str(int(e)) for e in self.exponents)
        return f"{sign_str}[{exp_str}]"

    def __eq__(self, other):
        if not isinstance(other, YANSNumber):
            return NotImplemented
        return self.sign == other.sign and self.exponents == other.exponents

    def __hash__(self):
        return hash((self.sign, self.exponents))

    def __pow__(self, exponent):
        """
        Raises the YANSNumber to a given exponent.
        Handles integer, rational, and real exponents.
        For negative base and non-integer exponent, returns complex result as (YANSNumber, phase).
        """
        try:
            exp_val = float(exponent)
        except (TypeError, ValueError):
            raise ValueError("Exponent must be numeric")

        if exp_val.is_integer():
            exp = int(exp_val)
            new_sign = self.sign if exp % 2 == 1 else 1
            new_exponents = [e * exp for e in self.exponents]
            return YANSNumber(new_sign, new_exponents)

        # Rational or real exponent
        new_exponents = [e * exp_val for e in self.exponents]

        # If base is negative and exponent is not integer, result is complex
        if self.sign == -1:
            phase = math.pi * exp_val
            return (YANSNumber(1, new_exponents), phase)
        else:
            return YANSNumber(1, new_exponents)

    def to_int(self):
        """
        Converts the YANSNumber back to its integer value.
        """
        if self.sign == 0:
            return 0
        if self.exponents == (0,):
            return self.sign
        primes = YANSNumber._get_primes_up_to_n(len(self.exponents))
        value = 1
        for p, e in zip(primes, self.exponents):
            if e != 0:
                value *= p ** e
        return self.sign * int(value)

    def to_base(self, base=10):
        """
        Returns the integer value as a string in the specified base.
        Supports bases up to 128 using ASCII characters.
        """
        if not 2 <= base <= 128:
            raise ValueError("Base must be between 2 and 128")
        n = abs(self.to_int())
        if self.sign == 0 or n == 0:
            return "0"
        charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}|;:,.<>?/`~"
        digits = []
        while n > 0:
            digits.append(charset[n % base])
            n //= base
        result = "".join(reversed(digits))
        return ('-' if self.sign == -1 else '') + result

    def __mul__(self, other):
        if not isinstance(other, YANSNumber):
            return NotImplemented
        if self.sign == 0 or other.sign == 0:
            return YANSNumber(0, [0])
        # Pad exponents to same length
        max_len = max(len(self.exponents), len(other.exponents))
        a = list(self.exponents) + [0] * (max_len - len(self.exponents))
        b = list(other.exponents) + [0] * (max_len - len(other.exponents))
        new_exponents = [x + y for x, y in zip(a, b)]
        new_sign = self.sign * other.sign
        return YANSNumber(new_sign, new_exponents)

    def __truediv__(self, other):
        if not isinstance(other, YANSNumber):
            return NotImplemented
        if other.sign == 0:
            raise ValueError("Division by zero YANSNumber")
        if self.sign == 0:
            return YANSNumber(0, [0])
        # Pad exponents to same length
        max_len = max(len(self.exponents), len(other.exponents))
        a = list(self.exponents) + [0] * (max_len - len(self.exponents))
        b = list(other.exponents) + [0] * (max_len - len(other.exponents))
        new_exponents = [x - y for x, y in zip(a, b)]
        new_sign = self.sign // other.sign  # 1/-1 division
        return YANSNumber(new_sign, new_exponents)

    def __abs__(self):
        return YANSNumber(abs(self.sign), self.exponents)

class YANSComplex:
    """
    Represents a complex number using two YANSNumber objects: real and imaginary parts.
    """
    def __init__(self, real, imag):
        if not isinstance(real, YANSNumber) or not isinstance(imag, YANSNumber):
            raise TypeError("real and imag must be YANSNumber instances")
        self.real = real
        self.imag = imag

    def __str__(self):
        real_str = str(self.real)
        imag_str = str(self.imag)
        if self.imag.sign == 0:
            return real_str
        sign = '+' if self.imag.sign == 1 else '-'
        imag_display = imag_str if self.imag.sign == 1 else imag_str[1:]
        return f"{real_str} {sign} {imag_display}i"

    def to_complex(self):
        """
        Converts to Python complex type.
        """
        return complex(self.real.to_int(), self.imag.to_int())

class YANSQuaternion:
    """
    Represents a quaternion using four YANSNumber objects: w + xi + yj + zk.
    """
    def __init__(self, w, x, y, z):
        if not all(isinstance(a, YANSNumber) for a in (w, x, y, z)):
            raise TypeError("All components must be YANSNumber instances")
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        parts = [
            str(self.w),
            f"{'+' if self.x.sign == 1 else '-'} {str(self.x)[1:] if self.x.sign == -1 else str(self.x)}i",
            f"{'+' if self.y.sign == 1 else '-'} {str(self.y)[1:] if self.y.sign == -1 else str(self.y)}j",
            f"{'+' if self.z.sign == 1 else '-'} {str(self.z)[1:] if self.z.sign == -1 else str(self.z)}k"
        ]
        return " ".join(parts)

    def to_tuple(self):
        return (self.w.to_int(), self.x.to_int(), self.y.to_int(), self.z.to_int())

class YANSClifford:
    """
    Represents a Clifford algebra element as a list of YANSNumber coefficients for basis blades.
    blades: list of strings representing basis (e.g. ['1', 'e1', 'e2', 'e12'])
    coeffs: list of YANSNumber objects, same length as blades
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
            if coeff.sign != 0:
                terms.append(f"{str(coeff)}{blade if blade != '1' else ''}")
        return " + ".join(terms) if terms else "0"

    def to_dict(self):
        return {blade: coeff.to_int() for blade, coeff in zip(self.blades, self.coeffs)}

def yans_representation(n):
    """
    Returns the YANS (Yet Another Number Scheme) representation of an integer n
    as a YANSNumber object with sign bit (1, -1, or 0) and exponents list.
    Example: str(yans_representation(12)) -> '+[2|1]'
             str(yans_representation(-12)) -> '-[2|1]'
             str(yans_representation(0)) -> '0'
    """
    if n == 0:
        return YANSNumber(0, [0])
    sign = 1 if n > 0 else -1
    abs_n = abs(n)
    if abs_n == 1:
        return YANSNumber(sign, [0])
    exponents = []
    primes_iter = sympy.primerange(2, abs_n + 1)
    for p in primes_iter:
        count = 0
        while abs_n % p == 0:
            abs_n //= p
            count += 1
        exponents.append(count)
        if abs_n == 1:
            break
    while exponents and exponents[-1] == 0:
        exponents.pop()
    return YANSNumber(sign, exponents)