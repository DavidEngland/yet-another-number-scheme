# Algebra of YANS Numbers

## YANS Representation

A YANS number is represented as an exponent vector:
$\mathbf{e} = (e_0, e_1, e_2, ..., e_k)$

Where:
- $e_0$ is the exponent for $-1$ (encodes sign: 0 for positive, 1 for negative)
- $e_1, e_2, ...$ are exponents for primes 2, 3, 5, ...

The integer value is:
$$
n = (-1)^{e_0} \cdot \prod_{i=1}^k p_i^{e_i}
$$
where $p_i$ is the $i$-th prime.

## Zero

Zero is represented as empty exponents (`[]`) or by convention as `[0]`.

## Multiplication

Given $A = \mathbf{e}_A$ and $B = \mathbf{e}_B$:
- Pad exponent vectors to equal length.
- Add exponents: $\mathbf{e}_C = \mathbf{e}_A + \mathbf{e}_B$

$$
A \times B = (e_{A,0} + e_{B,0}, e_{A,1} + e_{B,1}, \ldots, e_{A,k} + e_{B,k})
$$

Example:
- $6 = [0|1|1]$
- $10 = [0|1|0|1]$
- $6 \times 10 = [0|2|1|1] = 60$

## Division

Given $A = \mathbf{e}_A$ and $B = \mathbf{e}_B$:
- Pad exponent vectors to equal length.
- Subtract exponents: $\mathbf{e}_C = \mathbf{e}_A - \mathbf{e}_B$

$$
A / B = (e_{A,0} - e_{B,0}, e_{A,1} - e_{B,1}, \ldots, e_{A,k} - e_{B,k})
$$

Example:
- $60 = [0|2|1|1]$
- $6 = [0|1|1]$
- $60 / 6 = [0|1|0|1] = 10$

## Powers

For $A = \mathbf{e}$ and $r \in \mathbb{R}$:
- Multiply all exponents by $r$: $\mathbf{e}' = r \cdot \mathbf{e}$

$$
A^r = (r \cdot e_0, r \cdot e_1, \ldots, r \cdot e_k)
$$

Example:
- $6 = [0|1|1]$
- $6^2 = [0|2|2] = 36$

## Addition and Subtraction

Addition and subtraction are not closed in YANS representation.  
To add two YANS numbers, convert to integers, add, and convert back:

$$
\text{yans}(a) + \text{yans}(b) = \text{yans}(a_\text{int} + b_\text{int})
$$

## Complex YANS

A complex YANS is a pair of YANS numbers $(A, B)$ representing:
$$
z = A + Bi
$$

Multiplication follows complex number rules:
$$(A + Bi) \times (C + Di) = (AC - BD) + (AD + BC)i$$

## Quaternion YANS

A quaternion YANS is a quadruple of YANS numbers $(A, B, C, D)$ representing:
$$
q = A + Bi + Cj + Dk
$$

Multiplication follows quaternion rules with $i^2 = j^2 = k^2 = ijk = -1$.

## Clifford Algebra YANS

A Clifford algebra element is represented as a sum:
$$
M = \sum_{I} A_I e_I
$$
where $A_I$ are YANS numbers and $e_I$ are basis blades (e.g., 1, $e_1$, $e_2$, $e_{12}$, etc.).

Multiplication follows the geometric product rules, combining both inner (dot) and outer (wedge) products.

## Basis Elements

The YANS system has natural basis elements corresponding to the prime numbers:

- `[1]` = -1 (the sign unit)
- `[0|1]` = 2 (first prime)
- `[0|0|1]` = 3 (second prime) 
- `[0|0|0|1]` = 5 (third prime)
- `[0|0|0|0|1]` = 7 (fourth prime)
- And so on...

Any integer can be represented as a product of these basis elements raised to their respective powers. This gives YANS a natural vector space structure where each axis corresponds to a prime number (including -1 as the "zeroth prime").

For example, the number 30 = 2 × 3 × 5 has the representation `[0|1|1|1]`, which can be seen as the product of the basis elements:
`[0|1]` × `[0|0|1]` × `[0|0|0|1]` = `[0|1|1|1]`

## Properties

- Multiplication and division are closed and efficient in YANS.
- Exponentiation by integers is closed in YANS.
- Addition requires conversion to integers.
- The representation naturally captures the structure of numbers through their prime factorization.

## Symbolic Computation and Exact Arithmetic

One of the most powerful features of YANS is its ability to perform symbolic algebraic computations and delay numerical evaluation until needed:

1. **Exact Representation**: YANS stores numbers in their prime factorization form, which is exact (no floating-point errors).

2. **Algebraic Manipulation**: Operations can be performed symbolically on the exponent vectors:
   - Multiplication: add exponents
   - Division: subtract exponents
   - Powers: multiply exponents by the power
   - Roots: divide exponents by the root degree

3. **Delayed Evaluation**: The numerical value only needs to be calculated at the final step using `to_int()`.

### Example: Computing $(12^3 \div 8)^{1/2}$ exactly

```python
a = yans_representation(12)   # [0|2|1]
b = a ** 3                    # [0|6|3]
c = b / yans_representation(8)  # [0|3|3]
d = c ** 0.5                  # [0|1.5|1.5]
print(d.to_factor_string())   # 2^1.5 * 3^1.5
print(d.to_int())             # 36 (the exact answer)
```

This approach is particularly valuable for:
- Working with large integers where floating-point would lose precision
- Maintaining exact representations in algebraic expressions
- Detecting patterns in the factorization of results
- Avoiding intermediate rounding errors

YANS can function like a computer algebra system (CAS) for integer arithmetic, calculating symbolic results and only converting to numeric form when explicitly requested.

## Complex Exponentiation

Complex exponentiation like i^i is an interesting case that can be handled in YANS.

In mathematics, i^i is calculated as follows:
- i = e^(iπ/2)
- i^i = (e^(iπ/2))^i = e^(iπ/2 × i) = e^(-π/2) ≈ 0.2079...

This can be implemented in YANS by:

```python
from yans3 import YANSComplex, yans_representation

# Create i
zero = yans_representation(0)
one = yans_representation(1)
i = YANSComplex(zero, one)  # 0 + 1i

# Calculate i^i
# Method 1: Direct calculation
result = i.__pow__(i)  # Returns (real, imag) tuple: (0.20787957635076, 0)

# Method 2: Using Euler's formula
import math
pi = math.pi
e_neg_pi_half = math.exp(-pi/2)
result_yans = yans_representation(round(e_neg_pi_half))  # Approximate integer

# Note: For exact symbolic computation, you could represent:
# i^i = e^(-π/2)
# where e, π are treated as symbols in a more advanced CAS approach
```

This demonstrates how YANS can be extended to handle complex exponentiation and other transcendental operations, either numerically or symbolically.

## Gelfond-Schneider Theorem and Transcendental Numbers

YANS can also represent certain algebraic numbers (like √2) and can be extended to handle transcendental numbers through the Gelfond-Schneider Theorem, which states:

> If α and β are algebraic numbers with α ≠ 0, α ≠ 1, and β is irrational, then α^β is transcendental.

This theorem allows us to identify when expressions like a^b yield transcendental numbers. Some examples:

- 2^√2 is transcendental (base is rational, exponent is irrational)
- (√2)^(√2) is transcendental (base is irrational, exponent is irrational)
- π^e is undetermined by this theorem (both are transcendental)

The `gelfond_schneider.py` module extends YANS with:

1. Representations of algebraic numbers via their minimal polynomials
2. Detection of transcendental powers using the theorem
3. Symbolic computation with transcendental results

```python
from gelfond_schneider import AlgebraicNumber, compute_power

# Create algebraic numbers
sqrt2 = AlgebraicNumber.sqrt(2)
two = AlgebraicNumber.from_rational(yans_representation(2))

# Check if result is transcendental
result = compute_power(two, sqrt2)  # "Transcendental by Gelfond-Schneider: AlgNum([0|1])^AlgNum(poly=[-2, 0, 1], root=0)"

# Compute a power that's algebraic
result = compute_power(sqrt2, two)  # AlgNum([0|1]) (which is 2)
```

This extension allows YANS to handle a broader class of numbers and determine when operations yield transcendental results.
