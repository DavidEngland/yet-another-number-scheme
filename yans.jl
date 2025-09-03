module YANS

using Primes

"""
    YANSNumber
    
Represents an integer in YANS format: first exponent is for -1, 
followed by exponents for primes 2, 3, 5, etc.
"""
struct YANSNumber
    exponents::Vector{Int}
    
    # Constructors
    YANSNumber(exps::Vector{Int}) = new(exps)
    YANSNumber() = new([])  # Zero
end

# Utility functions
zero() = YANSNumber()
one() = YANSNumber([0])

# Operations are naturally vectorized in Julia
Base.:*(a::YANSNumber, b::YANSNumber) = begin
    max_len = max(length(a.exponents), length(b.exponents))
    a_padded = [a.exponents; zeros(Int, max(0, max_len - length(a.exponents)))]
    b_padded = [b.exponents; zeros(Int, max(0, max_len - length(b.exponents)))]
    YANSNumber(a_padded + b_padded)
end

# Clean conversion to integer with comprehensions
function to_int(n::YANSNumber)::BigInt
    isempty(n.exponents) && return 0  # Handle zero
    
    # First exponent is for -1, rest for primes
    result = BigInt((-1)^(n.exponents[1]))
    
    # Multiply by prime powers
    for (i, exp) in enumerate(n.exponents[2:end])
        p = nth_prime(i)
        result *= p^exp
    end
    
    return result
end

# More concise representation functions
# ...
end
