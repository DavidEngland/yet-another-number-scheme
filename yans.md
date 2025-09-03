YANS, which stands for **Y**et **A**nother **N**umber **S**cheme, is a method for representing integers using a list of exponents and a sign. It's essentially a way to encode the prime factorization of a number.

### How it Works

The fundamental idea behind YANS is to represent any non-zero integer $n$ by its unique prime factorization. The system uses a sign bit (1 for positive, -1 for negative) and a list of exponents corresponding to the prime numbers in ascending order (2, 3, 5, 7, etc.).

For a positive integer $n$, its prime factorization is $n = p_1^{e_1} \cdot p_2^{e_2} \cdot p_3^{e_3} \cdots$, where $p_i$ are the prime numbers in order and $e_i$ are their corresponding exponents. The YANS representation stores these exponents in a list. For example, to represent the number 12:

1.  **Prime Factorization:** The prime factorization of 12 is $2^2 \cdot 3^1$.
2.  **Exponents:** The exponent for the first prime (2) is 2, and the exponent for the second prime (3) is 1. All subsequent prime exponents are 0.
3.  **YANS Representation:** The YANS representation is a sign (+1) and the list of exponents [2, 1]. This is written as `+[2|1]`.

The class `YANSNumber` has two attributes:
* `sign`: An integer that is either 1 or -1.
* `exponents`: A list of integers representing the exponents of the primes.

For a number like -12, the sign is -1, and the exponents are the same as for 12, resulting in `-[2|1]`. A number with a large prime factor, like 17, is represented as `+[0|0|0|0|0|0|0]` because its prime factorization is just $17^1$. However, the `yans_representation` function correctly handles this by extending the list of primes as needed.

### Mathematical Operations

The YANS scheme makes multiplication, division, and exponentiation particularly straightforward. Since these operations on numbers correspond to simple additions or multiplications of their prime exponents, performing them in the YANS format can be computationally efficient.

The provided `__pow__` method demonstrates this. When raising a YANS number to an integer power, the new exponents are simply the old exponents multiplied by the power. For example, $(12)^3 = (2^2 \cdot 3^1)^3 = 2^6 \cdot 3^3$. In YANS, this is calculated by multiplying each exponent by 3: `+[2|1]` becomes `+[6|3]`. This is equivalent to $(+[2|1])^3$.

The `__pow__` method also handles non-integer exponents. For a positive YANS number, it scales the exponents by the floating-point value. For a negative YANS number raised to a non-integer power, the result is complex, and the method returns a tuple containing the new YANS number (with sign 1) and the phase angle in radians.  This phase is calculated as $\pi \times \text{exponent}$, based on the fact that raising -1 to a non-integer power $x$ results in $e^{i\pi x}$.