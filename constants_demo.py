"""
Demonstration of symbolic constants in YANS
"""
from yans4 import (
    YANSNumber, yans_representation, YANSSymbolic,
    PI, E, GOLDEN_RATIO, yans_pi, yans_e
)

def demo_constants():
    """Show how to use YANS constants."""
    # Create some basic YANS numbers
    two = yans_representation(2)
    three = yans_representation(3)
    
    # Create symbolic expressions with constants
    expr1 = YANSSymbolic(two, {PI: (YANSNumber.one(), 1)})  # 2π
    expr2 = YANSSymbolic(three, {E: (YANSNumber.one(), 2)})  # 3e²
    
    # Alternative creation with helpers
    pi_expr = yans_pi()  # π
    two_pi = YANSSymbolic(two)  # 2
    two_pi = two_pi * pi_expr  # 2π
    
    # Display exact symbolic forms
    print(f"Expression 1: {expr1}")  # 2 · π
    print(f"Expression 2: {expr2}")  # 3 · e²
    
    # Multiply symbolic expressions
    product = expr1 * expr2  # 6π·e²
    print(f"Product: {product}")
    
    # Convert to floating-point when needed
    print(f"Numeric value: {product.to_float():.10f}")  # ≈ 44.1332
    
    # Combine multiple constants
    combined = YANSSymbolic(two, {
        PI: (YANSNumber.one(), 1),
        E: (three, 2)
    })  # 2 · π · 3e²
    print(f"Combined: {combined}")
    print(f"Numeric value: {combined.to_float():.10f}")

if __name__ == "__main__":
    demo_constants()
