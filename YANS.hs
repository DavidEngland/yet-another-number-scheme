module YANS (
    YANSNumber,
    yansRepresentation,
    toInt,
    multiply,
    divide,
    power
) where

import Data.List (intercalate)
import Data.Numbers.Primes (primes, isPrime)

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

-- | Convert a YANS number to an integer
toInt :: YANSNumber -> Integer
toInt (YANSNumber []) = 0  -- Zero
toInt (YANSNumber (s:es)) = 
    let sign = if odd s then -1 else 1
        primePowers = zipWith (^) (take (length es) (tail primes)) es
    in sign * product primePowers

-- | Create YANS representation from an integer
yansRepresentation :: Integer -> YANSNumber
yansRepresentation 0 = YANSNumber []
yansRepresentation n = 
    let sign = if n < 0 then 1 else 0
        absN = abs n
        -- ... factorization logic ...
    in YANSNumber (sign : factorize absN)
