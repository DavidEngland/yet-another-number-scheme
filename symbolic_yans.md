# Symbolic Extensions for YANS

## Representing Mathematical Constants

While YANS excels at representing rational numbers and certain algebraic numbers (like √2), transcendental constants like π, e, and γ (Euler's constant) require special handling. This document proposes an extension to YANS that supports exact symbolic computations with these constants.

## Approach

We can extend YANS to handle mathematical constants by:

1. Creating a `YANSConstant` class to represent fundamental mathematical constants
2. Implementing a `YANSSymbolic` class that combines YANS numbers with constants
3. Maintaining exact symbolic representations until numerical evaluation is required

## Implementation Concept

```python
class YANSConstant:
    """Represents a mathematical constant like π or e."""
    def __init__(self, name, approximation_func=None, bbp_func=None):
        self.name = name  # 'pi', 'e', etc.
        self.approximation_func = approximation_func  # Function to compute approximation
        self.bbp_func = bbp_func  # Function to compute specific digits
    
    def __str__(self):
        return self.name

class YANSSymbolic:
    """
    Represents a symbolic expression combining YANS numbers and constants.
    Structure: {
        'yans': YANSNumber,         # Base YANS number
        'constants': {              # Dictionary of constants and their exponents
            constant_obj: exponent,  # e.g., {pi: 1, e: 2} represents π * e²
            ...
        }
    }
    """
    def __init__(self, yans_num, constants=None):
        self.yans = yans_num  # YANSNumber component
        self.constants = constants or {}  # Constants with exponents

    def __mul__(self, other):
        # Implementation to multiply symbolic expressions
        # Adds exponents for the same constants
        pass
    
    def __pow__(self, exponent):
        # Implementation to raise symbolic expression to a power
        # Multiplies all exponents by the power
        pass
    
    def to_float(self, precision=53):
        # Convert to floating-point approximation with given precision
        pass
```

## Common Mathematical Constants

```python
# Define common constants
PI = YANSConstant(
    'π', 
    approximation_func=math.pi,
    bbp_func=lambda d: bbp_pi(d)  # Bailey-Borwein-Plouffe formula
)

E = YANSConstant(
    'e',
    approximation_func=math.e
)

GAMMA = YANSConstant(
    'γ',
    approximation_func=0.57721566490153286
)
```

## Examples

### Exact Symbolic Computation

```python
# Create 2π
two = yans_representation(2)
two_pi = YANSSymbolic(two, {PI: 1})  # 2π

# Create e^(√2)
sqrt2 = YANSNumber([0, 0.5])  # √2 = 2^(1/2)
e_sqrt2 = YANSSymbolic(yans_representation(1), {E: sqrt2.to_float()})  # e^(√2)

# Multiply them: 2π * e^(√2)
result = two_pi * e_sqrt2  # Maintains exact symbolic form
```

### Numerical Evaluation

```python
# Get floating-point approximation (to default precision)
result.to_float()  # ≈ 24.0723...

# Get high-precision approximation (100 digits)
result.to_float(precision=100)
```

## BBP Formulas

For constants like π where BBP-type formulas exist, we can compute arbitrary hexadecimal (or decimal) digits without computing all previous digits:

```python
# Get the 1000th hexadecimal digit of π
PI.bbp_func(1000)  # Returns the digit at position 1000
```

## Extensions for Other Constants

This approach can be extended to other mathematical constants with known properties:

- Euler-Mascheroni constant (γ)
- Golden ratio (φ)
- Catalan's constant (G)
- Apéry's constant (ζ(3))

## Benefits

1. **Exact Representation**: Mathematical expressions involving constants remain exact.
2. **Symbolic Manipulation**: Operations like multiplication and exponentiation maintain symbolic form.
3. **Delayed Evaluation**: Numerical approximation happens only when explicitly requested.
4. **Precision Control**: When numerical values are needed, precision can be specified.

## Challenges

1. **Complex Expressions**: Managing complex expressions with multiple constants requires careful implementation.
2. **Simplification Rules**: Need rules for simplifying expressions (e.g., log(e^x) = x).
3. **Equality Testing**: Determining if two symbolic expressions are equal can be difficult.

This extension would significantly enhance YANS's ability to work with mathematical constants exactly, making it more powerful for applications in mathematical computing.
