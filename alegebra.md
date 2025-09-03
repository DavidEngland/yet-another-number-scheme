# Algebra of YANS Numbers

## YANS Representation

A YANS number is represented as a pair:
- Sign bit: $s \in \{-1, 0, +1\}$
- Exponent vector: $\mathbf{e} = (e_1, e_2, ..., e_k)$

The integer value is:
$$
n = s \cdot \prod_{i=1}^k p_i^{e_i}
$$
where $p_i$ is the $i$-th prime.

## Addition

Addition is not closed in YANS representation.  
To add two YANS numbers, convert to integers, add, and convert back:
$$
\text{yans}(a) + \text{yans}(b) = \text{yans}(a_\text{int} + b_\text{int})
$$

## Multiplication

Given $A = (s_A, \mathbf{e}_A)$ and $B = (s_B, \mathbf{e}_B)$:
- Pad exponent vectors to equal length.
- Multiply signs: $s_C = s_A \cdot s_B$
- Add exponents: $\mathbf{e}_C = \mathbf{e}_A + \mathbf{e}_B$

$$
A \times B = (s_A s_B, \; (e_{A,1} + e_{B,1}, \ldots, e_{A,k} + e_{B,k}))
$$

## Division

Given $A = (s_A, \mathbf{e}_A)$ and $B = (s_B, \mathbf{e}_B)$:
- Pad exponent vectors to equal length.
- Divide signs: $s_C = s_A / s_B$
- Subtract exponents: $\mathbf{e}_C = \mathbf{e}_A - \mathbf{e}_B$

$$
A / B = (s_A / s_B, \; (e_{A,1} - e_{B,1}, \ldots, e_{A,k} - e_{B,k}))
$$

## Powers

For $A = (s, \mathbf{e})$ and $r \in \mathbb{R}$:
- Exponents: $\mathbf{e}' = r \cdot \mathbf{e}$
- Sign: $s' = s$ if $r$ is odd integer, $+1$ otherwise
- If $s = -1$ and $r$ not integer, result is complex:
  $$(A^r = (\prod p_i^{r e_i}, \text{phase} = \pi r))$$

## Absolute Value

$$
|A| = (|s|, \mathbf{e})
$$

## Complex YANS

A complex YANS is $(A, B)$, where $A$ and $B$ are YANS numbers:
$$
Z = A + Bi
$$

## Quaternion YANS

A quaternion YANS is $(A, B, C, D)$:
$$
Q = A + Bi + Cj + Dk
$$

## Clifford Algebra YANS

A Clifford algebra element is a sum:
$$
C = \sum_{I} A_I e_I
$$
where $A_I$ are YANS numbers and $e_I$ are basis blades.

## Zero

Zero is represented as $(0, [0])$.

## Notes

- Multiplication and division are closed in YANS.
- Addition is not closed; conversion to integer is required.
- Exponent vectors may be rational or real for roots and powers.
