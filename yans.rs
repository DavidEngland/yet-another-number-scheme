use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Zero, Signed};
use primal::Sieve;
use std::fmt;

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
    
    /// One representation
    pub fn one() -> Self {
        Self { exponents: vec![0] } // First exponent 0 = positive, no -1 factor
    }
    
    /// Convert to integer value
    pub fn to_int(&self) -> BigInt {
        // Handle empty exponents (zero)
        if self.exponents.is_empty() {
            return BigInt::zero();
        }
        
        // Get sign from first exponent (-1 raised to exponent power)
        let sign = if self.exponents[0] % 2 == 1 {
            BigInt::from(-1)
        } else {
            BigInt::from(1)
        };
        
        // If only sign exponent is present, return just the sign
        if self.exponents.len() == 1 {
            return sign;
        }
        
        // Calculate required sieve size based on number of exponents
        let sieve_size = (self.exponents.len() * 10).max(1000);
        let sieve = Sieve::new(sieve_size);
        
        let mut result = BigInt::one();
        
        // Multiply by prime powers, starting from second exponent
        for (i, &exp) in self.exponents.iter().enumerate().skip(1) {
            if exp == 0 {
                continue;
            }
            
            // Get i-th prime (note: i is 0-based after skip(1))
            match sieve.nth_prime(i) {
                Some(prime) => {
                    let prime_big = BigInt::from(prime);
                    result *= prime_big.pow(exp as u32);
                },
                None => {
                    panic!("Prime at index {} is beyond sieve capacity of {}", i, sieve_size);
                }
            }
        }
        
        // Apply sign
        sign * result
    }
    
    /// Get prime factorization as a string
    pub fn to_factor_string(&self) -> String {
        if self.exponents.is_empty() {
            return "0".to_string();
        }
        
        if self.exponents.len() == 1 && self.exponents[0] == 0 {
            return "1".to_string();
        }
        
        let mut factors = Vec::new();
        
        // Handle -1 factor
        if !self.exponents.is_empty() && self.exponents[0] % 2 == 1 {
            factors.push("-1".to_string());
        }
        
        // Calculate required sieve size
        let sieve_size = (self.exponents.len() * 10).max(1000);
        let sieve = Sieve::new(sieve_size);
        
        // Add prime factors with exponents
        for (i, &exp) in self.exponents.iter().enumerate().skip(1) {
            if exp == 0 {
                continue;
            }
            
            match sieve.nth_prime(i) {
                Some(prime) => {
                    if exp == 1 {
                        factors.push(format!("{}", prime));
                    } else {
                        factors.push(format!("{}^{}", prime, exp));
                    }
                },
                None => {
                    factors.push(format!("p_{}^{}", i, exp)); // Placeholder when prime is unknown
                }
            }
        }
        
        if factors.is_empty() {
            "1".to_string()
        } else {
            factors.join(" * ")
        }
    }
}

/// Multiplication implementation
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

/// Division implementation
impl std::ops::Div for YANSNumber {
    type Output = Self;
    
    fn div(self, other: Self) -> Self {
        if other.exponents.is_empty() {
            panic!("Division by zero");
        }
        
        let max_len = self.exponents.len().max(other.exponents.len());
        let mut result = vec![0; max_len];
        
        for i in 0..max_len {
            let a = if i < self.exponents.len() { self.exponents[i] } else { 0 };
            let b = if i < other.exponents.len() { other.exponents[i] } else { 0 };
            result[i] = a - b;
        }
        
        // Trim trailing zeros
        while !result.is_empty() && *result.last().unwrap() == 0 {
            result.pop();
        }
        
        Self { exponents: result }
    }
}

/// Display implementation
impl fmt::Display for YANSNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.exponents.is_empty() {
            return write!(f, "[0]");
        }
        
        write!(f, "[{}]", self.exponents.iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("|"))
    }
}

/// Convert from BigInt to YANSNumber
pub fn yans_representation(n: &BigInt) -> YANSNumber {
    // Handle zero
    if n.is_zero() {
        return YANSNumber::zero();
    }
    
    // Handle sign
    let sign_exp = if n.is_negative() { 1 } else { 0 };
    let abs_n = n.abs();
    
    // Handle 1 and -1
    if abs_n == BigInt::one() {
        return YANSNumber::new(vec![sign_exp]);
    }
    
    let mut exponents = vec![sign_exp];
    let mut remainder = abs_n.clone();
    
    // Trial division with small primes first
    let mut sieve = Sieve::new(1000);
    let mut prime_idx = 0;
    
    while remainder > BigInt::one() {
        // Ensure we have enough primes
        if prime_idx >= sieve.prime_pi() {
            // Grow the sieve
            sieve = Sieve::new(sieve.upper_bound() * 2);
            if prime_idx >= sieve.prime_pi() {
                // Handle extremely large primes by appending a placeholder
                exponents.push(1); // This prime appears once
                break;
            }
        }
        
        let p = sieve.nth_prime(prime_idx).unwrap();
        let p_big = BigInt::from(p);
        
        let mut count = 0;
        
        // Extract all factors of this prime
        while (&remainder % &p_big).is_zero() {
            remainder = remainder / &p_big;
            count += 1;
        }
        
        exponents.push(count);
        
        // Move to next prime
        prime_idx += 1;
        
        // If remainder is 1, we're done
        if remainder == BigInt::one() {
            break;
        }
        
        // Check if remainder is prime by trial division
        let sqrt_remainder = remainder.sqrt();
        let mut is_prime = true;
        
        for i in 0..prime_idx {
            let p = sieve.nth_prime(i).unwrap();
            let p_big = BigInt::from(p);
            if p_big > sqrt_remainder {
                break;
            }
            if (&remainder % &p_big).is_zero() {
                is_prime = false;
                break;
            }
        }
        
        // If remainder is prime, add it and we're done
        if is_prime {
            // Add zeros for missing prime exponents
            while exponents.len() <= prime_idx + 1 {
                exponents.push(0);
            }
            exponents[exponents.len() - 1] = 1;
            break;
        }
    }
    
    // Trim trailing zeros
    while !exponents.is_empty() && *exponents.last().unwrap() == 0 {
        exponents.pop();
    }
    
    YANSNumber::new(exponents)
}

// Implement From<BigInt> for YANSNumber
impl From<BigInt> for YANSNumber {
    fn from(value: BigInt) -> Self {
        yans_representation(&value)
    }
}

// Tests would go here
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_zero() {
        let zero = YANSNumber::zero();
        assert_eq!(zero.to_int(), BigInt::zero());
    }
    
    #[test]
    fn test_one() {
        let one = YANSNumber::one();
        assert_eq!(one.to_int(), BigInt::one());
    }
    
    #[test]
    fn test_to_int() {
        // Test 12 = 2² * 3¹
        let twelve = YANSNumber::new(vec![0, 2, 1]);
        assert_eq!(twelve.to_int(), BigInt::from(12));
        
        // Test -12 = -1¹ * 2² * 3¹
        let neg_twelve = YANSNumber::new(vec![1, 2, 1]);
        assert_eq!(neg_twelve.to_int(), BigInt::from(-12));
    }
    
    #[test]
    fn test_multiplication() {
        let six = YANSNumber::new(vec![0, 1, 1]);
        let ten = YANSNumber::new(vec![0, 1, 0, 1]);
        
        let product = six.clone() * ten.clone();
        assert_eq!(product.to_int(), BigInt::from(60));
    }
}
