# Finding the N-th Prime Number

In YANS, getting the n-th prime number efficiently is important for both conversion and representation. Here are several approaches with their trade-offs.

## 1. Using sympy (Python)

The `sympy` library offers a straightforward way to find the n-th prime:

```python
import sympy

def nth_prime(n):
    """Return the n-th prime number (0-indexed)."""
    if n < 0:
        raise ValueError("Index must be non-negative")
    return sympy.prime(n + 1)  # sympy uses 1-indexing
```

This method is clean and accurate but can be slow for large n.

## 2. Cached Prime List (Python)

For repeated access, maintaining a cached list of primes improves performance:

```python
class PrimeCache:
    def __init__(self):
        self._primes = [2, 3, 5, 7, 11, 13]  # Initial cache
    
    def nth_prime(self, n):
        """Get the n-th prime (0-indexed)."""
        while n >= len(self._primes):
            self._extend_cache()
        return self._primes[n]
    
    def _extend_cache(self):
        """Find the next prime and add to cache."""
        candidate = self._primes[-1] + 2
        while True:
            if all(candidate % p != 0 for p in self._primes if p * p <= candidate):
                self._primes.append(candidate)
                return
            candidate += 2

# Usage
prime_cache = PrimeCache()
print(prime_cache.nth_prime(100))  # 541
```

## 3. Prime Number Theorem Estimation

For very large n, the Prime Number Theorem provides an approximation:

```python
import math

def estimate_nth_prime(n):
    """Estimate the n-th prime using the Prime Number Theorem."""
    if n < 5:
        return [2, 3, 5, 7, 11][n]
    # PNT: p_n ≈ n * log(n) + n * log(log(n))
    return int(n * (math.log(n) + math.log(math.log(n))))
```

This estimate gets more accurate as n increases.

## 4. Sieve-Based Approach (Rust)

For high-performance applications, using a sieve in Rust:

```rust
use primal::Sieve;

fn nth_prime(n: usize) -> Option<usize> {
    let estimated_upper_bound = if n < 5 {
        12
    } else {
        // Approximate upper bound using PNT
        let float_n = n as f64;
        (float_n * (float_n.ln() + float_n.ln().ln())) as usize
    };
    
    let sieve = Sieve::new(estimated_upper_bound);
    sieve.nth_prime(n)
}
```

## 5. Implementation in YANS

For our YANS library, we can integrate prime number access directly:

```python
# Add to yans3.py
def nth_prime(n):
    """Get the n-th prime number (0-indexed)."""
    if n == 0:
        return 2  # First prime
    
    # Use cached class attribute
    while n >= len(YANSNumber._primes):
        YANSNumber._primes.append(sympy.nextprime(YANSNumber._primes[-1]))
    
    return YANSNumber._primes[n]
```

## Performance Comparison

| Method                | Speed             | Memory Use        | Accuracy          |
|-----------------------|-------------------|-------------------|-------------------|
| sympy.prime()         | Medium            | Low               | Perfect           |
| Cached list           | Fast for repeated | High for large n  | Perfect           |
| PNT estimation        | Very fast         | None              | Approximate       |
| Sieve (Rust)          | Very fast         | Medium            | Perfect           |

For YANS applications, a hybrid approach using a growing cache with periodic sieving provides the best balance of performance and accuracy.