module YANS (
    YANSNumber,
    yansRepresentation,
    toInt,
    multiply,
    divide,
    power
) where

import Data.List (intercalate)
import Data.Numbers.Primes (primes, isPrime, factorize)

-- | Represents a number in YANS format
newtype YANSNumber = YANSNumber { exponents :: [Int] }
    deriving (Eq)

instance Show YANSNumber where
    show (YANSNumber []) = "[0]"  -- Zero
    show (YANSNumber es) = "[" ++ intercalate "|" (map show es) ++ "]"

-- | Multiply two YANS numbers
multiply :: YANSNumber -> YANSNumber -> YANSNumber
multiply (YANSNumber a) (YANSNumber b) = YANSNumber $ zipWithDefault 0 0 (+) a b
  where
    zipWithDefault :: a -> a -> (a -> a -> a) -> [a] -> [a] -> [a]
    zipWithDefault defA defB f as bs = 
        let len = max (length as) (length bs)
            padA = as ++ replicate (len - length as) defA
            padB = bs ++ replicate (len - length bs) defB
        in zipWith f padA padB

-- | Divide two YANS numbers
divide :: YANSNumber -> YANSNumber -> YANSNumber
divide (YANSNumber a) (YANSNumber b) = YANSNumber $ zipWithDefault 0 0 (-) a b

-- | Raise a YANS number to a power
power :: YANSNumber -> Int -> YANSNumber
power (YANSNumber []) _ = YANSNumber []  -- 0^n = 0
power _ 0 = YANSNumber [0]  -- n^0 = 1
power (YANSNumber es) n = YANSNumber $ map (* n) es

-- | Convert a YANS number to an integer
toInt :: YANSNumber -> Integer
toInt (YANSNumber []) = 0  -- Zero
toInt (YANSNumber (s:es)) = 
    let sign = if odd s then -1 else 1
        -- Use the infinite primes list, but zip with the finite exponents list
        primePowers = map (\(p, e) -> p ^ e) $ zip (tail primes) es
    in sign * product primePowers

-- | Create YANS representation from an integer
yansRepresentation :: Integer -> YANSNumber
yansRepresentation 0 = YANSNumber []
yansRepresentation n = 
    let sign = if n < 0 then 1 else 0
        absN = abs n
        factors = factorizeToList absN
    in YANSNumber (sign : factors)

-- | Convert prime factorization to exponent list
factorizeToList :: Integer -> [Int]
factorizeToList 1 = []
factorizeToList n =
    let factors = Data.Numbers.Primes.factorize n
        primeList = uniqueSorted $ map fst factors
        maxPrime = last primeList
        primesNeeded = length $ takeWhile (<= maxPrime) primes
        
        -- Create a map from prime to index
        primeIndices = zip (tail primes) [0..]
        
        -- Initialize result list with zeros
        result = replicate primesNeeded 0
        
        -- Update result with actual exponents
        updateExponents [] acc = acc
        updateExponents ((p, e):rest) acc =
            case lookup p primeIndices of
                Just idx -> updateExponents rest (setAt idx e acc)
                Nothing -> error "Prime not found in index list"
    in updateExponents factors result
  where
    -- Helper functions
    uniqueSorted = foldl (\acc x -> if x `elem` acc then acc else acc ++ [x]) []
    setAt idx val list = take idx list ++ [val] ++ drop (idx + 1) list

-- Make YANSNumber an instance of standard typeclasses
instance Semigroup YANSNumber where
    (<>) = multiply

instance Monoid YANSNumber where
    mempty = YANSNumber [0]  -- Identity element for multiplication is 1

-- Num instance for standard arithmetic operations
instance Num YANSNumber where
    -- Multiplication is direct
    (*) = multiply
    
    -- Addition and subtraction require conversion to integers and back
    a + b = yansRepresentation (toInt a + toInt b)
    a - b = yansRepresentation (toInt a - toInt b)
    
    -- Absolute value changes sign exponent to even
    abs (YANSNumber []) = YANSNumber []
    abs (YANSNumber (s:es)) = YANSNumber (s `mod` 2 : es)
    
    -- Signum returns -1, 0, or 1 based on sign
    signum (YANSNumber []) = YANSNumber []
    signum (YANSNumber (s:_)) = if odd s 
                               then YANSNumber [1]  -- -1
                               else YANSNumber [0]  -- +1
    
    -- Convert integer to YANS
    fromInteger = yansRepresentation
