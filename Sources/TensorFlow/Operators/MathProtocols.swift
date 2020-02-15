//
//  MathProtocols.swift
//  Experimental
//
//  Created by Artem Artemev on 19/01/2020.
//



public typealias Real = Float  // Temporary plug


public protocol ShapedCollection: RandomAccessCollection, MutableCollection {
    associatedtype Scalar

    /// The number of dimensions of the array.
    var rank: Int { get }
    /// The shape of the array.
    var shape: [Int] { get }
    /// The total number of scalars in the array.
    var scalarCount: Int { get }

    /// Creates an array with the specified shape and contiguous scalars in row-major order.
    /// - Precondition: The number of scalars must equal the product of the dimensions of the shape.
    init(shape: [Int], scalars: [Scalar])

    /// Creates an array with the specified shape and sequence of scalars in row-major order.
    /// - Precondition: The number of scalars must equal the product of the dimensions of the shape.
    init<S: Sequence>(shape: [Int], scalars: S) where S.Element == Scalar
}

public protocol Arithmetics {
    mutating func add(_ other: Self)
    
    mutating func sub(_ other: Self)
    
    mutating func mul(_ other: Self)
    
    static func + (lhs: Self, rhs: Self) -> Self

    static func - (lhs: Self, rhs: Self) -> Self

    static func * (lhs: Self, rhs: Self) -> Self
}

extension Arithmetics {
    static func + (lhs: Self, rhs: Self) -> Self {
        var copy = lhs
        copy.add(rhs)
        return copy
    }
    
    static func - (lhs: Self, rhs: Self) -> Self {
        var copy = lhs
        copy.sub(rhs)
        return copy
    }

    static func * (lhs: Self, rhs: Self) -> Self {
        var copy = lhs
        copy.mul(rhs)
        return copy
    }
}


public protocol BasicFunctions {
    static func sqrt(_ x: Self) -> Self

    /// The cosine of `x`, interpreted as an angle in radians.
    static func cos(_ x: Self) -> Self

    /// The sine of `x`, interpreted as an angle in radians.
    static func sin(_ x: Self) -> Self

    /// The tangent of `x`, interpreted as an angle in radians.
    static func tan(_ x: Self) -> Self

    /// The inverse cosine of `x` in radians.
    static func acos(_ x: Self) -> Self

    /// The inverse sine of `x` in radians.
    static func asin(_ x: Self) -> Self

    /// The inverse tangent of `x` in radians.
    static func atan(_ x: Self) -> Self

    /// The hyperbolic cosine of `x`.
    static func cosh(_ x: Self) -> Self

    /// The hyperbolic sine of `x`.
    static func sinh(_ x: Self) -> Self

    /// The hyperbolic tangent of `x`.
    static func tanh(_ x: Self) -> Self

    /// The inverse hyperbolic cosine of `x`.
    static func acosh(_ x: Self) -> Self

    /// The inverse hyperbolic sine of `x`.
    static func asinh(_ x: Self) -> Self

    /// The inverse hyperbolic tangent of `x`.
    static func atanh(_ x: Self) -> Self

    /// The exponential function applied to `x`, or `e**x`.
    static func exp(_ x: Self) -> Self

    /// Two raised to to power `x`.
    static func exp2(_ x: Self) -> Self

    /// Ten raised to to power `x`.
    static func exp10(_ x: Self) -> Self

    /// `exp(x) - 1` evaluated so as to preserve accuracy close to zero.
    static func expm1(_ x: Self) -> Self
    
    /// The natural logarithm of `x`.
    static func log(_ x: Self) -> Self

    /// The base-two logarithm of `x`.
    static func log2(_ x: Self) -> Self

    /// The base-ten logarithm of `x`.
    static func log10(_ x: Self) -> Self

    /// `log(1 + x)` evaluated so as to preserve accuracy close to zero.
    static func log1p(_ x: Self) -> Self

    /// `exp(y log(x))` computed without loss of intermediate precision.
    ///
    /// For real types, if `x` is negative the result is NaN, even if `y` has
    /// an integral value. For complex types, there is a branch cut on the
    /// negative real axis.
    static func pow(_ x: Self, _ y: Self) -> Self

    /// `x` raised to the `n`th power.
    ///
    /// The product of `n` copies of `x`.
    static func pow(_ x: Self, _ n: Int) -> Self

    /// The `n`th root of `x`.
    ///
    /// For real types, if `x` is negative and `n` is even, the result is NaN.
    /// For complex types, there is a branch cut along the negative real axis.
    static func root(_ x: Self, _ n: Int) -> Self
}

public protocol RandomShapedCollection {
    init(randomNormal: [Int], mean: Real, std: Real)
    init(randomUniform: [Int], min: Real, max: Real)
    func choice(size: Int, withReplacement replace: Bool) -> Self
    func shuffle() -> Self
}

public protocol ShapedCollectionManipulations {
// List of functions:
//    reshape
//    bandPart
//    diagonal
//    diagonalPart
//    transpose
//    permute
//    roll
}

public protocol LinearAlgebra:
        Arithmetics,
        BasicFunctions,
        ShapedCollectionManipulations {
// List of functions:
//    cholesky
//    solve
//    triangularSolve
//    det
//    logdet
//    slogdet
//    trace
//    svd
//    matmul
//    matvec
//    eye
//    qr
}

extension BasicFunctions where Self: Differentiable {
    @inlinable
    @differentiable
    static func sqrt(_ x: Self) -> Self {
        .sqrt(x)
    }
    
    @inlinable
    @derivative(of: sqrt)
    static public func _vjpSqrt(_ x: Self) -> (value: Self, pullback: (Self) -> Self) {
        let value = self.sqrt(x)
        return (value, { v in v / (2.0 * value) })
    }
}

extension Tensor: BasicFunctionsShapedCollection where Scalar: TensorFlowFloatingPoint {
    @inlinable
    @differentiable
    static public func function(_ x: Self) -> Self {
        .sqrt(x)
    }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
    @inlinable
    @derivative(of: function)
    static public func _vjpFunction(_ x: Self) -> (value: Self, pullback: (Self) -> Self) {
        (x, {x in x})
    }
    
}
