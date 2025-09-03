# Yet Another Number Scheme (YANS)

A Python implementation of YANS (Yet Another Number Scheme), an efficient representation of numbers based on prime factorization.

## What is YANS?

YANS represents integers as an ordered list of exponents for their prime factorization. The first exponent represents the power of -1 (for sign), followed by exponents for primes 2, 3, 5, 7, etc.

For example:
- 12 = 2² × 3¹ is represented as `[0|2|1]` (the 0 means "positive")
- -12 = -1¹ × 2² × 3¹ is represented as `[1|2|1]` (the 1 means "negative")

## Key Features

- **Efficient Operations**: Multiplication, division, and powers become simple vector operations
- **Zero Handling**: Special representation for zero
- **Type Hints**: Clean, typed implementations for maintainability
- **Advanced Math**: Extended to complex numbers, quaternions, and Clifford algebra

## Usage Examples

### Basic Integer Representation

```python
from yans3 import yans_representation

# Basic representation
print(yans_representation(12))      # [0|2|1]
print(yans_representation(-12))     # [1|2|1]
print(yans_representation(0))       # [0]

# Convert back to integer
yans_num = yans_representation(42)
print(yans_num.to_int())            # 42

# Show factorization
print(yans_num.to_factor_string())  # 2^1 * 3^1 * 7^1
```

### Operations

```python
a = yans_representation(6)    # [0|1|1]
b = yans_representation(10)   # [0|1|0|1]

# Multiplication: 6 × 10 = 60
c = a * b                     # [0|2|1|1]
print(c.to_int())             # 60

# Division: 60 ÷ 6 = 10
d = c / a                     # [0|1|0|1]
print(d.to_int())             # 10

# Powers: 6² = 36
e = a ** 2                    # [0|2|2]
print(e.to_int())             # 36
```

### Complex Numbers and Beyond

```python
from yans3 import YANSComplex, yans_representation

# Complex number: 3 + 4i
real = yans_representation(3)
imag = yans_representation(4)
z = YANSComplex(real, imag)

# Operations
z_squared = z * z             # -7 + 24i
```

### Blade-Based Clifford Algebra (Geometric Algebra)

```python
from bladed_yans import Multivector, yans

# Create vectors
e1 = Multivector.basis_vector(1)
e2 = Multivector.basis_vector(2)

# Geometric product
bivector = e1 * e2            # e12 (represents an oriented plane)

# Rotation using rotors
angle = math.pi/4
rotor = Multivector.exp(bivector * yans_representation(angle))
```

## Versions

- **yans.py**: Original implementation
- **yans2.py**: Extended to complex numbers and quaternions
- **yans3.py**: Improved implementation with better type handling and zero support
- **bladed_yans.py**: Implementation using geometric algebra concepts (blades)

## Documentation

See additional files for detailed documentation:
- [YANS Description](yans.md): Detailed explanation of the YANS representation
- [Algebra](algebra.md): Mathematical properties of YANS numbers
- [Blades](blades.md): Explanation of geometric algebra blades and their implementation
- [Applications](applications.md): Potential applications of YANS

## License

MIT

