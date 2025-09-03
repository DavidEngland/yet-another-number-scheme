import sympy
import math

class YANSNumber:
    def __init__(self, sign, exponents):
        self.sign = sign  # 1 or -1
        self.exponents = exponents  # list of ints

    def __str__(self):
        sign_str = '+' if self.sign == 1 else '-'
        exp_str = "|".join(str(e) for e in self.exponents)
        return f"{sign_str}[{exp_str}]"

    def __pow__(self, exponent):
        """
        Raises the YANSNumber to a given exponent.
        Handles integer, rational, and real exponents.
        For negative base and non-integer exponent, returns complex result as (YANSNumber, phase).
        """
        # Integer exponent
        if isinstance(exponent, int) or (isinstance(exponent, float) and exponent.is_integer()):
            exp = int(exponent)
            new_sign = self.sign if exp % 2 == 1 else 1
            new_exponents = [e * exp for e in self.exponents]
            return YANSNumber(new_sign, new_exponents)
        # Rational or real exponent
        try:
            exp_val = float(exponent)
        except Exception:
            raise ValueError("Exponent must be numeric")
        new_exponents = [e * exp_val for e in self.exponents]
        # If base is negative and exponent is not integer, result is complex
        if self.sign == -1:
            # phase = pi * exponent
            phase = math.pi * exp_val
            # Return (YANSNumber, phase) tuple for quick/compact storage
            return (YANSNumber(1, new_exponents), phase)
        else:
            return YANSNumber(1, new_exponents)

    def to_int(self):
        """
        Converts the YANSNumber back to its integer value.
        """
        if self.exponents == [0]:
            return self.sign
        primes = list(sympy.primerange(2, 2 + len(self.exponents)))
        value = 1
        for p, e in zip(primes, self.exponents):
            value *= p ** e
        return self.sign * value

    def to_base(self, base=10):
        """
        Returns the integer value as a string in the specified base.
        Supports bases up to 128 using ASCII characters.
        """
        n = abs(self.to_int())
        if base < 2 or base > 128:
            raise ValueError("Base must be between 2 and 128")
        # Character set for base conversion (0-9, a-z, A-Z, and more)
        charset = (
            '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
            '!@#$%^&*()-_=+[]{}|;:,.<>?/`~"\'\\'
        )
        # Pad charset if base > len(charset)
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
        return ('-' if self.sign == -1 else '') + digits

def yans_representation(n):
    """
    Returns the YANS (Yet Another Number Scheme) representation of an integer n
    as a YANSNumber object with sign bit (1 or -1) and exponents list.
    Example: str(yans_representation(12)) -> '+[2|1]'
             str(yans_representation(-12)) -> '-[2|1]'
    """
    if n == 0:
        raise ValueError("n must be nonzero")
    sign = 1 if n > 0 else -1
    abs_n = abs(n)
    if abs_n == 1:
        return YANSNumber(sign, [0])
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
    return YANSNumber(sign, exponents)