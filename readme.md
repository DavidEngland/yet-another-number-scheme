# Yet Another Number Scheme (YANS)

This repository provides a Python implementation of the YANS (Yet Another Number Scheme) representation for integers.

## What is YANS?

YANS represents an integer as an ordered, bracketed, and delimited list of exponents in its prime factorization.  
For example, the number 12 (which is 2^2 * 3^1) is represented as `[2|1]`.

## Usage

```python
from yans import yans_representation

print(yans_representation(12))  # Output: [2|1]
print(yans_representation(1))   # Output: [0]
print(yans_representation(30))  # Output: [1|1|1]
```

## Function Documentation

### `yans_representation(n: int) -> str`

Returns the YANS representation of a positive integer `n` as a string.  
The format is `[e1|e2|...|ek]`, where each `ei` is the exponent of the i-th prime in the factorization of `n`.

- For `n = 1`, returns `[0]`.
- For composite numbers, uses primes up to `sqrt(n)` and includes any remaining prime factor > `sqrt(n)`.

## Example Table: YANS and Base 36 Representations

| Integer | YANS Representation | Base 36 |
|---------|---------------------|---------|
| 50      | +[1|2]              | 1e      |
| 51      | +[0|1|1|1]          | 1f      |
| 52      | +[2|0|1]            | 1g      |
| 53      | +[0|0|0|0|1]        | 1h      |
| 54      | +[1|3]              | 1i      |
| 55      | +[0|1|0|1]          | 1j      |
| 56      | +[3|1]              | 1k      |
| 57      | +[0|0|1|1]          | 1l      |
| 58      | +[1|0|0|1]          | 1m      |
| 59      | +[0|0|0|0|0|1]      | 1n      |
| 60      | +[2|1|1]            | 1o      |
| ...     | ...                 | ...     |
| 198     | +[1|1|1|2]          | 5o      |
| 199     | +[0|0|0|0|0|0|1]    | 5p      |
| 200     | +[3|0|0|1]          | 5q      |

*Table truncated for brevity. See script below to generate full table.*

### How to generate this table

```python
from yans2 import yans_representation

print("| Integer | YANS Representation | Base 36 |")
print("|---------|---------------------|---------|")
for i in range(50, 201):
    yans = yans_representation(i)
    print(f"| {i} | {str(yans)} | {yans.to_base(36)} |")
```

## License

MIT

