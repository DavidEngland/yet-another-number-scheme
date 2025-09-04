#!/usr/bin/env python3
from bernoulli import YANSRational, BernoulliNumbers
from yans4 import yans_representation

def test_specific_bernoulli():
    bernoulli = BernoulliNumbers()
    
    # Check B12 (important for irregular prime analysis)
    b12 = bernoulli.get(12)
    print(f"B_12 = {b12}")
    
    # Examine numerator in YANS format
    print(f"Numerator: {b12.numerator}")
    print(f"Prime factorization: {b12.numerator.to_factor_string()}")
    
    # For example, 691 is a known irregular prime dividing B12
    print(f"691 divides B12 numerator: {b12.numerator.to_int() % 691 == 0}")

def test_custom_rational():
    # Create custom rational numbers using YANS
    num1 = YANSRational(yans_representation(22), yans_representation(7))
    num2 = YANSRational(yans_representation(355), yans_representation(113))
    
    # These are approximations of π
    print(f"22/7 = {num1.to_fraction().numerator/num1.to_fraction().denominator}")
    print(f"355/113 = {num2.to_fraction().numerator/num2.to_fraction().denominator}")
    
    # Multiply these approximations
    product = num1 * num2
    print(f"Product: {product}")
    print(f"As float: {product.to_fraction().numerator/product.to_fraction().denominator}")

def verify_kummer_congruence():
    """Test Kummer's congruence for Bernoulli numbers"""
    b = BernoulliNumbers()
    
    # Kummer's congruence for a prime p says:
    # B_{p-3} ≡ -3B_{p-5} (mod p) for p ≥ 5
    
    p = 11  # Choose a prime
    b1 = b.get(p-3).to_fraction()
    b2 = -3 * b.get(p-5).to_fraction()
    
    print(f"B_{p-3} = {b1}")
    print(f"-3*B_{p-5} = {b2}")
    print(f"Difference: {b1 - b2}")
    print(f"Congruent modulo {p}: {(b1.numerator * b2.denominator - b2.numerator * b1.denominator) % (p * b1.denominator * b2.denominator) == 0}")

if __name__ == "__main__":
    print("=== Testing specific Bernoulli numbers ===")
    test_specific_bernoulli()
    
    print("\n=== Testing custom rational numbers ===")
    test_custom_rational()
    
    print("\n=== Verifying Kummer's congruence ===")
    verify_kummer_congruence()
