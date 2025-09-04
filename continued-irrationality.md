# Continued Fractions and Irrationality Measures

## 1. Introduction to Continued Fractions

A continued fraction is a representation of a real number using a sequence of integers and recursive fractions. The general form is:

$$a_0 + \cfrac{1}{a_1 + \cfrac{1}{a_2 + \cfrac{1}{a_3 + \ddots}}}$$

We typically denote this as $[a_0; a_1, a_2, a_3, \ldots]$.

### 1.1 Simple Examples

- $\pi = [3; 7, 15, 1, 292, 1, \ldots]$
- $e = [2; 1, 2, 1, 1, 4, 1, \ldots]$
- $\sqrt{2} = [1; 2, 2, 2, 2, \ldots]$ (periodic)
- $\phi = [1; 1, 1, 1, 1, \ldots]$ (golden ratio)

### 1.2 Finite vs. Infinite

- **Finite continued fractions** represent rational numbers
- **Infinite continued fractions** represent irrational numbers
- **Periodic continued fractions** represent quadratic irrationals (solutions to $ax^2 + bx + c = 0$ with $a,b,c \in \mathbb{Z}$)

### 1.3 Convergents

The sequence of rational approximations obtained by truncating a continued fraction:

For $[a_0; a_1, a_2, \ldots, a_n]$, we get convergents $\frac{p_n}{q_n}$ where:

$$p_0 = a_0, \quad p_1 = a_1a_0 + 1, \quad p_n = a_np_{n-1} + p_{n-2}$$
$$q_0 = 1, \quad q_1 = a_1, \quad q_n = a_nq_{n-1} + q_{n-2}$$

## 2. Computing Continued Fractions

### 2.1 Algorithm for Converting a Number to a Continued Fraction

```python
def to_continued_fraction(x: float, max_terms: int = 10) -> list:
    """Convert a real number to its continued fraction representation."""
    cf = []
    for _ in range(max_terms):
        a = int(x)  # Integer part
        cf.append(a)
        frac = x - a  # Fractional part
        if abs(frac) < 1e-10:  # Stop if we've reached a rational
            break
        x = 1 / frac  # Reciprocal of fractional part
    return cf
```

### 2.2 Algorithm for Converting a Continued Fraction to a Number

```python
def from_continued_fraction(cf: list) -> float:
    """Convert a continued fraction back to a floating-point number."""
    result = 0
    for a in reversed(cf):
        result = a + (1 / result if result != 0 else 0)
    return result
```

### 2.3 Integration with YANS Framework

With the ExactCompute framework, we can store continued fractions exactly:

```python
class ContinuedFraction:
    def __init__(self, terms: List[int]):
        self.terms = terms
        
    def to_exact_number(self) -> ExactNumber:
        """Convert to an ExactNumber representation."""
        result = ExactNumber(0)
        for a in reversed(self.terms):
            result = ExactNumber(a) + (ExactNumber(1) / result if result != 0 else 0)
        return result
```

## 3. Irrationality Measure

The irrationality measure μ(α) of a real number α is the supremum of the set of real numbers μ such that:

$$\left|\alpha - \frac{p}{q}\right| < \frac{1}{q^\mu}$$

has infinitely many integer solutions p, q with q > 0.

### 3.1 Significance

- Rational numbers have irrationality measure 1
- Almost all irrational numbers have irrationality measure 2 (Khintchine's theorem)
- Algebraic numbers have finite irrationality measure (Roth's theorem)
- Transcendental numbers can have irrationality measure 2 or higher
- Liouville numbers have infinite irrationality measure

### 3.2 Known Values

- All rational numbers: μ = 1
- Almost all irrationals: μ = 2
- Algebraic numbers (degree ≥ 2): 2 ≤ μ ≤ d (where d is the degree)
- e: μ = 2
- π: μ = 2 (conjectured, proven ≤ 7.6063)
- Liouville's constant: μ = ∞

## 4. Estimating Irrationality Measure

### 4.1 Using Continued Fractions

The irrationality measure of a number can be estimated from the growth rate of its continued fraction terms:

```python
def estimate_irrationality_measure(cf: list, window: int = 5) -> float:
    """
    Estimate irrationality measure using continued fraction.
    A rough estimate based on the growth rate of terms.
    """
    if len(cf) < window + 1:
        return float('nan')  # Not enough terms
        
    # Examine growth of convergent denominators
    q = [1, cf[1]]
    for i in range(2, len(cf)):
        q.append(cf[i] * q[i-1] + q[i-2])
    
    # Calculate growth rate of approximation error
    errors = []
    for i in range(window, len(cf)):
        prev_error = abs(from_continued_fraction(cf[:i-1]) - from_continued_fraction(cf))
        curr_error = abs(from_continued_fraction(cf[:i]) - from_continued_fraction(cf))
        
        # Approximation: |α - p/q| ≈ 1/q^μ
        # So μ ≈ -log(error)/log(q)
        if prev_error > 0 and curr_error > 0:
            errors.append(-math.log(curr_error)/math.log(q[i]))
    
    return sum(errors)/len(errors) if errors else float('nan')
```

### 4.2 Using Diophantine Approximation

A more direct approach is to find good rational approximations and measure how quickly the error decreases:

```python
def irrationality_measure_diophantine(x: float, max_q: int = 10000) -> float:
    """
    Estimate irrationality measure using Diophantine approximation.
    Find the best μ such that |x - p/q| < 1/q^μ has many solutions.
    """
    approximations = []
    
    for q in range(1, max_q):
        p = round(x * q)
        error = abs(x - p/q)
        approximations.append((p, q, error))
    
    # Sort by error (best approximations first)
    approximations.sort(key=lambda x: x[2])
    
    # Select best approximations that represent local minima
    best_approx = []
    for i, (p, q, error) in enumerate(approximations[:100]):
        if i == 0 or q > best_approx[-1][1]:
            best_approx.append((p, q, error))
    
    # Calculate μ for each approximation
    measures = []
    for p, q, error in best_approx:
        # From |x - p/q| < 1/q^μ
        # We get μ ≈ -log(error)/log(q)
        if error > 0:
            measures.append(-math.log(error)/math.log(q))
    
    return sum(measures)/len(measures) if measures else float('nan')
```

## 5. Historical Context and Development

### 5.1 Cantor's Work

Georg Cantor, though primarily known for his work on set theory and transfinite numbers, also contributed to the study of irrational numbers. His approach to classifying real numbers led to fundamental insights about different "sizes" of infinity, including:

- Proving that transcendental numbers exist
- Showing that almost all real numbers are transcendental
- Developing the concept of "uncountability"

### 5.2 Development of Irrationality Measure

- **Joseph Liouville (1844)**: Introduced the concept and constructed the first examples of transcendental numbers with infinite irrationality measure (Liouville numbers)
- **Axel Thue (1909)**: Proved that algebraic numbers of degree ≥ 3 have irrationality measure < 2
- **Klaus Roth (1955)**: Proved that all algebraic numbers of degree ≥ 2 have irrationality measure exactly 2 (Fields Medal work)
- **W.M. Schmidt (1970s)**: Extended to simultaneous approximation (Schmidt's subspace theorem)

### 5.3 Modern Applications

- **Cryptography**: Numbers with high irrationality measures create strong pseudorandom sequences
- **Dynamical Systems**: The irrationality measure affects how orbits distribute in certain dynamical systems
- **Diophantine Approximation**: Critical in solving equations in integers

## 6. Implementation in YANS Framework

### 6.1 Continued Fraction Representation

```python
def yans_to_continued_fraction(num: YANSNumber, max_terms: int = 20) -> List[int]:
    """Convert a YANS number to its continued fraction representation."""
    x = num.to_int()
    return to_continued_fraction(x, max_terms)

def continued_fraction_to_yans(cf: List[int]) -> YANSNumber:
    """Convert a continued fraction to a YANS number (rational approximation)."""
    p, q = [0, 1], [1, 0]
    for i, a in enumerate(cf):
        p.append(a * p[i+1] + p[i])
        q.append(a * q[i+1] + q[i])
    return yans_representation(p[-1]) / yans_representation(q[-1])
```

### 6.2 Irrationality Analyzer

```python
class IrrationalityAnalyzer:
    """Analyze the irrationality properties of numbers in the YANS framework."""
    
    @staticmethod
    def analyze(num: Union[float, YANSNumber], max_terms: int = 50) -> Dict[str, Any]:
        """Analyze the irrationality properties of a number."""
        if isinstance(num, YANSNumber):
            x = num.to_int()
        else:
            x = num
            
        cf = to_continued_fraction(x, max_terms)
        periodic = detect_periodicity(cf)
        
        result = {
            "continued_fraction": cf,
            "periodicity": periodic,
            "quadratic_irrational": periodic is not None,
            "convergents": calculate_convergents(cf),
            "estimated_measure": None
        }
        
        # Only estimate measure for non-periodic (likely transcendental)
        if not periodic:
            result["estimated_measure"] = estimate_irrationality_measure(cf)
            
        return result
        
    @staticmethod
    def is_liouville_number(x: float, depth: int = 10) -> bool:
        """
        Test if a number might be a Liouville number.
        A number is Liouville if it has arbitrarily good rational approximations.
        """
        cf = to_continued_fraction(x, depth)
        # Look for exceptionally large terms
        return any(term > 10**6 for term in cf)
```

## 7. Conclusion and Further Research Directions

The study of continued fractions and irrationality measures provides deep insights into the nature of real numbers. Within the YANS framework, these tools can be used to:

1. **Identify Number Classes**: Distinguish between rational, algebraic irrational, and transcendental numbers
2. **Optimize Approximations**: Find the best rational approximations for calculations
3. **Detect Patterns**: Uncover underlying patterns in numerical data
4. **Construct Special Numbers**: Create numbers with specific properties

Future research could focus on:

- Developing more efficient algorithms for computing continued fractions of special constants
- Exploring the connection between irrationality measures and computational complexity
- Using machine learning to predict patterns in continued fraction expansions
- Extending the framework to p-adic numbers and other number systems

---

## References

1. Khinchin, A. Y. (1964). *Continued Fractions*. University of Chicago Press.
2. Bugeaud, Y. (2004). *Approximation by Algebraic Numbers*. Cambridge University Press.
3. Shallit, J. (1979). "Simple continued fractions for some irrational numbers." *Journal of Number Theory*, 11(2), 209-217.
4. Borwein, J., & Bailey, D. (2008). *Mathematics by Experiment: Plausible Reasoning in the 21st Century*. A K Peters/CRC Press.
