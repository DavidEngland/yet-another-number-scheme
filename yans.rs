use num_bigint::BigInt;
use num_traits::{One, Zero};
use primal::Sieve;

/// Represents a number in YANS format.
#[derive(Clone, Debug, PartialEq)]
pub struct YANSNumber {
    /// Exponents for primes, starting with -1
    exponents: Vec<i32>,
}

impl YANSNumber {
    /// Create a new YANS number from exponents
    pub fn new(exponents: Vec<i32>) -> Self {
        Self { exponents }
    }
    
    /// Zero representation
    pub fn zero() -> Self {
        Self { exponents: vec![] }
    }
    
    /// Convert to integer value
    pub fn to_int(&self) -> BigInt {
        if self.exponents.is_empty() {
            return BigInt::zero();
        }
        
        let mut result = if self.exponents[0] % 2 == 1 {
            BigInt::from(-1)
        } else {
            BigInt::from(1)
        };
        
        let sieve = Sieve::new(1000); // Cache primes
        
        for (i, &exp) in self.exponents.iter().enumerate().skip(1) {
            if exp == 0 {
                continue;
            }
            let prime = sieve.nth_prime(i);
            let prime_big = BigInt::from(prime);
            result *= prime_big.pow(exp as u32);
        }
        
        result
    }
}

// Implement multiplication via the Mul trait
impl std::ops::Mul for YANSNumber {
    type Output = Self;
    
    fn mul(self, other: Self) -> Self {
        let max_len = self.exponents.len().max(other.exponents.len());
        let mut result = vec![0; max_len];
        
        for i in 0..max_len {
            let a = if i < self.exponents.len() { self.exponents[i] } else { 0 };
            let b = if i < other.exponents.len() { other.exponents[i] } else { 0 };
            result[i] = a + b;
        }
        
        Self { exponents: result }
    }
}

// Additional implementations...
