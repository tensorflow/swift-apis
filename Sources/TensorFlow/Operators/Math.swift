// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

infix operator .> : ComparisonPrecedence
infix operator .== : ComparisonPrecedence

// TODO:
// - Consider explicit broadcasting for elementwise binary ops when
//   scalarization and rank getter are implemented.

// TODO: Remove the following extension once `./` and `./=` are defined for 
// `PointwiseMultiplicative`.

infix operator ./ : MultiplicationPrecedence
infix operator ./= : AssignmentPrecedence

public extension PointwiseMultiplicative {
    static func ./ (lhs: Self, rhs: Self) -> Self {
        lhs .* rhs.reciprocal
    }

    static func ./= (lhs: inout Self, rhs: Self) {
        lhs = lhs ./ rhs
    }
}

//===------------------------------------------------------------------------------------------===//
// Generic Elementary Functions
//===------------------------------------------------------------------------------------------===//

extension Tensor: ElementaryFunctions where Scalar: TensorFlowFloatingPoint {
    /// The square root of `x`.
    ///
    /// For real types, if `x` is negative the result is `.nan`. For complex
    /// types there is a branch cut on the negative real axis.
    @differentiable
    public static func sqrt(_ x: Self) -> Self {
        TensorFlow.sqrt(x)
    }

    /// The cosine of `x`, interpreted as an angle in radians.
    @differentiable
    public static func cos(_ x: Self) -> Self {
        TensorFlow.cos(x)
    }

    /// The sine of `x`, interpreted as an angle in radians.
    @differentiable
    public static func sin(_ x: Self) -> Self {
        TensorFlow.sin(x)
    }

    /// The tangent of `x`, interpreted as an angle in radians.
    @differentiable
    public static func tan(_ x: Self) -> Self {
        TensorFlow.tan(x)
    }

    /// The inverse cosine of `x` in radians.
    @differentiable
    public static func acos(_ x: Self) -> Self {
        TensorFlow.acos(x)
    }

    /// The inverse sine of `x` in radians.
    @differentiable
    public static func asin(_ x: Self) -> Self {
        TensorFlow.asin(x)
    }

    /// The inverse tangent of `x` in radians.
    @differentiable
    public static func atan(_ x: Self) -> Self {
        TensorFlow.atan(x)
    }

    /// The hyperbolic cosine of `x`.
    @differentiable
    public static func cosh(_ x: Self) -> Self {
        TensorFlow.cosh(x)
    }

    /// The hyperbolic sine of `x`.
    @differentiable
    public static func sinh(_ x: Self) -> Self {
        TensorFlow.sinh(x)
    }

    /// The hyperbolic tangent of `x`.
    @differentiable
    public static func tanh(_ x: Self) -> Self {
        TensorFlow.tanh(x)
    }

    /// The inverse hyperbolic cosine of `x`.
    @differentiable
    public static func acosh(_ x: Self) -> Self {
        TensorFlow.acosh(x)
    }

    /// The inverse hyperbolic sine of `x`.
    @differentiable
    public static func asinh(_ x: Self) -> Self {
        TensorFlow.asinh(x)
    }

    /// The inverse hyperbolic tangent of `x`.
    @differentiable
    public static func atanh(_ x: Self) -> Self {
        TensorFlow.atanh(x)
    }

    /// The exponential function applied to `x`, or `e**x`.
    @differentiable
    public static func exp(_ x: Self) -> Self {
        TensorFlow.exp(x)
    }

    /// Two raised to to power `x`.
    @differentiable
    public static func exp2(_ x: Self) -> Self {
        TensorFlow.exp2(x)
    }

    /// Ten raised to to power `x`.
    @differentiable
    public static func exp10(_ x: Self) -> Self {
        TensorFlow.exp10(x)
    }

    /// `exp(x) - 1` evaluated so as to preserve accuracy close to zero.
    @differentiable
    public static func expm1(_ x: Self) -> Self {
        TensorFlow.expm1(x)
    }

    /// The natural logarithm of `x`.
    @differentiable
    public static func log(_ x: Self) -> Self {
        TensorFlow.log(x)
    }

    /// The base-two logarithm of `x`.
    @differentiable
    public static func log2(_ x: Self) -> Self {
        TensorFlow.log2(x)
    }

    /// The base-ten logarithm of `x`.
    @differentiable
    public static func log10(_ x: Self) -> Self {
        TensorFlow.log10(x)
    }

    /// `log(1 + x)` evaluated so as to preserve accuracy close to zero.
    @differentiable
    public static func log1p(_ x: Self) -> Self {
        TensorFlow.log1p(x)
    }

    /// `exp(y log(x))` computed without loss of intermediate precision.
    ///
    /// For real types, if `x` is negative the result is NaN, even if `y` has
    /// an integral value. For complex types, there is a branch cut on the
    /// negative real axis.
    @differentiable
    public static func pow(_ x: Self, _ y: Self) -> Self {
        TensorFlow.pow(x, y)
    }

    /// `x` raised to the `n`th power.
    ///
    /// The product of `n` copies of `x`.
    @differentiable
    public static func pow(_ x: Self, _ n: Int) -> Self {
        TensorFlow.pow(x, n)
    }

    /// The `n`th root of `x`.
    ///
    /// For real types, if `x` is negative and `n` is even, the result is NaN.
    /// For complex types, there is a branch cut along the negative real axis.
    @differentiable
    public static func root(_ x: Self, _ n: Int) -> Self {
        TensorFlow.root(x, n)
    }
}

//===------------------------------------------------------------------------------------------===//
// Vector Space
//===------------------------------------------------------------------------------------------===//

extension Tensor: VectorProtocol where Scalar: TensorFlowFloatingPoint {
    public typealias VectorSpaceScalar = Float

    // @differentiable(where Scalar: TensorFlowFloatingPoint)
    public func scaled(by scale: Float) -> Self {
        Scalar(scale) * self
    }

    // @differentiable(where Scalar: TensorFlowFloatingPoint)
    public func adding(_ scalar: Float) -> Self {
        self + Scalar(scalar)
    }

    // @differentiable(where Scalar: TensorFlowFloatingPoint)
    public func subtracting(_ scalar: Float) -> Self {
        self - Scalar(scalar)
    }
}

extension VectorProtocol {
    static func + (lhs: VectorSpaceScalar, rhs: Self) -> Self {
        rhs.adding(lhs)
    }

    static func + (lhs: Self, rhs: VectorSpaceScalar) -> Self {
        lhs.adding(rhs)
    }

    static func - (lhs: Self, rhs: VectorSpaceScalar) -> Self {
        lhs.subtracting(rhs)
    }

    static func * (lhs: VectorSpaceScalar, rhs: Self) -> Self {
        rhs.scaled(by: lhs)
    }

    static func * (lhs: Self, rhs: VectorSpaceScalar) -> Self {
        lhs.scaled(by: rhs)
    }
}

extension VectorProtocol where VectorSpaceScalar: SignedNumeric {
    static prefix func - (x: Self) -> Self {
        .zero - x
    }

    static func - (lhs: VectorSpaceScalar, rhs: Self) -> Self {
        (-rhs).adding(lhs)
    }
}

//===------------------------------------------------------------------------------------------===//
// Additional Element-wise Operators
//===------------------------------------------------------------------------------------------===//

public extension Tensor where Scalar: Numeric {
    /// Adds the scalar to every scalar of the tensor and produces the sum.
    @inlinable
    @differentiable(vjp: _vjpAdd(lhs:rhs:) where Scalar: TensorFlowFloatingPoint)
    static func + (lhs: Scalar, rhs: Tensor) -> Tensor {
        return Tensor(lhs) + rhs
    }

    /// Adds the scalar to every scalar of the tensor and produces the sum.
    @inlinable
    @differentiable(vjp: _vjpAdd(lhs:rhs:) where Scalar: TensorFlowFloatingPoint)
    static func + (lhs: Tensor, rhs: Scalar) -> Tensor {
        return lhs + Tensor(rhs)
    }

    /// Subtracts the scalar from every scalar of the tensor and produces the difference.
    @inlinable
    @differentiable(vjp: _vjpSubtract(lhs:rhs:) where Scalar: TensorFlowFloatingPoint)
    static func - (lhs: Scalar, rhs: Tensor) -> Tensor {
        return Tensor(lhs) - rhs
    }

    /// Subtracts the scalar from every scalar of the tensor and produces the difference
    @inlinable
    @differentiable(vjp: _vjpSubtract(lhs:rhs:) where Scalar: TensorFlowFloatingPoint)
    static func - (lhs: Tensor, rhs: Scalar) -> Tensor {
        return lhs - Tensor(rhs)
    }

    /// Adds two tensors and stores the result in the left-hand-side variable.
    /// - Note: `+=` supports broadcasting.
    @inlinable
    static func += (lhs: inout Tensor, rhs: Tensor) {
        lhs = lhs + rhs
    }

    /// Adds the scalar to every scalar of the tensor and stores the result in the left-hand-side
    /// variable.
    @inlinable
    static func += (lhs: inout Tensor, rhs: Scalar) {
        lhs = lhs + rhs
    }

    /// Subtracts the second tensor from the first and stores the result in the left-hand-side
    /// variable.
    /// - Note: `-=` supports broadcasting.
    @inlinable
    static func -= (lhs: inout Tensor, rhs: Tensor) {
        lhs = lhs - rhs
    }

    /// Subtracts the scalar from every scalar of the tensor and stores the result in the
    /// left-hand-side variable.
    @inlinable
    static func -= (lhs: inout Tensor, rhs: Scalar) {
        lhs = lhs - rhs
    }

    /// Returns the tensor produced by multiplying the two tensors.
    /// - Note: `*` supports broadcasting.
    @inlinable
    @differentiable(vjp: _vjpMultiply(lhs:rhs:) where Scalar: TensorFlowFloatingPoint)
    static func * (lhs: Tensor, rhs: Tensor) -> Tensor {
        return _Raw.mul(lhs, rhs)
    }

    /// Returns the tensor by multiplying it with every scalar of the tensor.
    @inlinable
    @differentiable(vjp: _vjpMultiply(lhs:rhs:) where Scalar: TensorFlowFloatingPoint)
    static func * (lhs: Scalar, rhs: Tensor) -> Tensor {
        return Tensor(lhs) * rhs
    }

    /// Multiplies the scalar with every scalar of the tensor and produces the product.
    @inlinable
    @differentiable(vjp: _vjpMultiply(lhs:rhs:) where Scalar: TensorFlowFloatingPoint)
    static func * (lhs: Tensor, rhs: Scalar) -> Tensor {
        return lhs * Tensor(rhs)
    }

    /// Multiplies two tensors and stores the result in the left-hand-side variable.
    /// - Note: `*=` supports broadcasting.
    @inlinable
    static func *= (lhs: inout Tensor, rhs: Tensor) {
        lhs = lhs * rhs
    }

    /// Multiplies the tensor with the scalar, broadcasting the scalar, and stores the result in the
    /// left-hand-side variable.
    @inlinable
    static func *= (lhs: inout Tensor, rhs: Scalar) {
        lhs = lhs * rhs
    }

    /// Returns the quotient of dividing the first tensor by the second.
    /// - Note: `/` supports broadcasting.
    @inlinable
    @differentiable(vjp: _vjpDivide(lhs:rhs:) where Scalar: TensorFlowFloatingPoint)
    static func / (lhs: Tensor, rhs: Tensor) -> Tensor {
        return _Raw.div(lhs, rhs)
    }

    /// Returns the quotient of dividing the scalar by the tensor, broadcasting the scalar.
    @inlinable
    @differentiable(vjp: _vjpDivide(lhs:rhs:) where Scalar: TensorFlowFloatingPoint)
    static func / (lhs: Scalar, rhs: Tensor) -> Tensor {
        return Tensor(lhs) / rhs
    }

    /// Returns the quotient of dividing the tensor by the scalar, broadcasting the scalar.
    @inlinable
    @differentiable(vjp: _vjpDivide(lhs:rhs:) where Scalar: TensorFlowFloatingPoint)
    static func / (lhs: Tensor, rhs: Scalar) -> Tensor {
        return lhs / Tensor(rhs)
    }

    /// Divides the first tensor by the second and stores the quotient in the left-hand-side
    /// variable.
    @inlinable
    static func /= (lhs: inout Tensor, rhs: Tensor) {
        lhs = lhs / rhs
    }

    /// Divides the tensor by the scalar, broadcasting the scalar, and stores the quotient in the
    /// left-hand-side variable.
    @inlinable
    static func /= (lhs: inout Tensor, rhs: Scalar) {
        lhs = lhs / rhs
    }

    /// Returns the remainder of dividing the first tensor by the second.
    /// - Note: `%` supports broadcasting.
    @inlinable
    static func % (lhs: Tensor, rhs: Tensor) -> Tensor {
        return _Raw.mod(lhs, rhs)
    }

    /// Returns the remainder of dividing the tensor by the scalar, broadcasting the scalar.
    @inlinable
    static func % (lhs: Tensor, rhs: Scalar) -> Tensor {
        return lhs % Tensor(rhs)
    }

    /// Returns the remainder of dividing the scalar by the tensor, broadcasting the scalar.
    @inlinable
    static func % (lhs: Scalar, rhs: Tensor) -> Tensor {
        return Tensor(lhs) % rhs
    }

    /// Divides the first tensor by the second and stores the remainder in the left-hand-side
    /// variable.
    @inlinable
    static func %= (lhs: inout Tensor, rhs: Tensor) {
        lhs = lhs % rhs
    }

    /// Divides the tensor by the scalar and stores the remainder in the left-hand-side variable.
    @inlinable
    static func %= (lhs: inout Tensor, rhs: Scalar) {
        lhs = lhs % rhs
    }
}

internal extension Tensor where Scalar: TensorFlowFloatingPoint {
    @inlinable
    static func _vjpAdd(lhs: Tensor, rhs: Scalar) -> (Tensor, (Tensor) -> (Tensor, Scalar)) {
        return (lhs + rhs, { v in (v, v.sum().scalarized()) })
    }

    @inlinable
    static func _vjpAdd(lhs: Scalar, rhs: Tensor) -> (Tensor, (Tensor) -> (Scalar, Tensor)) {
        return (lhs + rhs, { v in (v.sum().scalarized(), v) })
    }

    @inlinable
    static func _vjpSubtract(lhs: Tensor, rhs: Scalar) -> (Tensor, (Tensor) -> (Tensor, Scalar)) {
        return (lhs - rhs, { v in (v, -v.sum().scalarized()) })
    }

    @inlinable
    static func _vjpSubtract(lhs: Scalar, rhs: Tensor) -> (Tensor, (Tensor) -> (Scalar, Tensor)) {
        return (lhs - rhs, { v in (v.sum().scalarized(), -v) })
    }

    @inlinable
    static func _vjpMultiply(lhs: Tensor, rhs: Tensor) -> (Tensor, (Tensor) -> (Tensor, Tensor)) {
        return (lhs * rhs, { [lhsShape = lhs.shapeTensor, rhsShape = rhs.shapeTensor] v in
            let lhsGrad = rhs * v
            let rhsGrad = lhs * v
            let (lhsAxes, rhsAxes) = _Raw.broadcastGradientArgs(s0: lhsShape, s1: rhsShape)
            return (lhsGrad.sum(squeezingAxes: lhsAxes).reshaped(toShape: lhsShape),
                    rhsGrad.sum(squeezingAxes: rhsAxes).reshaped(toShape: rhsShape))
        })
    }

    @inlinable
    static func _vjpMultiply(lhs: Tensor, rhs: Scalar) -> (Tensor, (Tensor) -> (Tensor, Scalar)) {
        return (lhs * rhs, { v in (v * rhs, (v * lhs).sum().scalarized()) })
    }

    @inlinable
    static func _vjpMultiply(lhs: Scalar, rhs: Tensor) -> (Tensor, (Tensor) -> (Scalar, Tensor)) {
        return (lhs * rhs, { v in ((v * rhs).sum().scalarized(), v * lhs) })
    }

    @inlinable
    static func _vjpDivide(lhs: Tensor, rhs: Tensor) -> (Tensor, (Tensor) -> (Tensor, Tensor)) {
        return (lhs / rhs, { [lhsShape = lhs.shapeTensor, rhsShape = rhs.shapeTensor] v in
            let lhsGrad = v / rhs
            let rhsGrad = -lhs / rhs.squared() * v
            let (lhsAxes, rhsAxes) = _Raw.broadcastGradientArgs(s0: lhsShape, s1: rhsShape)
            return (lhsGrad.sum(squeezingAxes: lhsAxes).reshaped(toShape: lhsShape),
                    rhsGrad.sum(squeezingAxes: rhsAxes).reshaped(toShape: rhsShape))
        })
    }

    @inlinable
    static func _vjpDivide(lhs: Tensor, rhs: Scalar) -> (Tensor, (Tensor) -> (Tensor, Scalar)) {
        return (lhs / rhs, { v in
            (v / rhs, (v * -lhs / Tensor(rhs).squared()).sum().scalarized())
        })
    }

    @inlinable
    static func _vjpDivide(lhs: Scalar, rhs: Tensor) -> (Tensor, (Tensor) -> (Scalar, Tensor)) {
        return (lhs / rhs, { v in ((v / rhs).sum().scalarized(), v * -lhs / rhs.squared()) })
    }
}

public extension Tensor where Scalar == Bool {
    /// Returns `!self` element-wise.
    @inlinable
    func elementsLogicalNot() -> Tensor {
        return _Raw.logicalNot(self)
    }

    /// Returns `self && other` element-wise.
    /// - Note: `&&` supports broadcasting.
    @inlinable
    func elementsLogicalAnd(_ other: Tensor) -> Tensor {
        return _Raw.logicalAnd(self, other)
    }

    /// Returns `self && other` element-wise, broadcasting `other`.
    @inlinable
    func elementsLogicalAnd(_ other: Scalar) -> Tensor {
        return elementsLogicalAnd(Tensor(other))
    }

    /// Returns `self || other` element-wise.
    @inlinable
    func elementsLogicalOr(_ other: Tensor) -> Tensor {
        return _Raw.logicalOr(self, other)
    }

    /// Returns `self || other` element-wise, broadcasting `other`.
    @inlinable
    func elementsLogicalOr(_ other: Scalar) -> Tensor {
        return elementsLogicalOr(Tensor(other))
    }
}

public extension Tensor where Scalar: TensorFlowNumeric {
    /// Returns `max(min(self, max), min)`.
    @inlinable
    @differentiable(vjp: _vjpClipped where Scalar: TensorFlowFloatingPoint)
    func clipped(min: Tensor, max: Tensor) -> Tensor {
        _Raw.clipByValue(t: self, clipValueMin: min, clipValueMax: max)
    }

    /// Returns `max(min(self, max), min)`.
    @inlinable
    @differentiable(wrt: (self, min) where Scalar: TensorFlowFloatingPoint)
    func clipped(min: Tensor, max: Scalar) -> Tensor {
        clipped(min: min, max: Tensor(max))
    }

    /// Returns `max(min(self, max), min)`.
    @inlinable
    @differentiable(wrt: (self, max) where Scalar: TensorFlowFloatingPoint)
    func clipped(min: Scalar, max: Tensor) -> Tensor {
        clipped(min: Tensor(min), max: max)
    }

    /// Returns `max(min(self, max), min)`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func clipped(min: Scalar, max: Scalar) -> Tensor {
        clipped(min: Tensor(min), max: Tensor(max))
    }
}

internal extension Tensor where Scalar: TensorFlowFloatingPoint {
    @inlinable
    func _vjpClipped(min: Tensor, max: Tensor) -> (Tensor, (Tensor) -> (Tensor, Tensor, Tensor)) {
        (clipped(min: min, max: max), { v in
            let selfShape = self.shapeTensor
            let minShape = min.shapeTensor
            let maxShape = max.shapeTensor
            let zeros = Tensor(zerosLike: v)
            let minMask = self .< min
            let maxMask = self .> max
            let selfGradient = v.replacing(with: zeros, where: minMask.elementsLogicalOr(maxMask))
            let minGradient = zeros.replacing(with: v, where: minMask)
            let maxGradient = zeros.replacing(with: v, where: maxMask)
            let (selfAxes, minAxes) = _Raw.broadcastGradientArgs(s0: selfShape, s1: minShape)
            let (_, maxAxes) = _Raw.broadcastGradientArgs(s0: selfShape, s1: maxShape)
            return (selfGradient.sum(squeezingAxes: selfAxes).reshaped(toShape: selfShape),
                    minGradient.sum(squeezingAxes: minAxes).reshaped(toShape: minShape),
                    maxGradient.sum(squeezingAxes: maxAxes).reshaped(toShape: maxShape))
        })
    }
}

//===------------------------------------------------------------------------------------------===//
// Element-wise Unary Math Functions
//===------------------------------------------------------------------------------------------===//

// Export Glibc/Darwin math functions. We should not require users to import
// Foundation/Darwin/Glibc in order to use scalar math functions.
//
#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
@_exported import Darwin.C
#else
@_exported import Glibc
#endif
//
// FIXME(rxwei): Scoped imports are not yet supported in parseable module
// interfaces, so `@_exported import` won't work. When that becomes supported,
// switch to `@_exported import` by removing `import Darwin.C/Glibc` above and
// uncommenting the following lines. In the meantime, consider using indirect
// wrappers for each function so that random libc symbols won't be leaked to
// users' code completion.
//
// #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
// @_exported import func Darwin.C.sin
// @_exported import func Darwin.C.cos
// @_exported import func Darwin.C.tan
// @_exported import func Darwin.C.sinf
// @_exported import func Darwin.C.cosf
// @_exported import func Darwin.C.tanf
// @_exported import func Darwin.C.sinh
// @_exported import func Darwin.C.cosh
// @_exported import func Darwin.C.tanh
// @_exported import func Darwin.C.sinhf
// @_exported import func Darwin.C.coshf
// @_exported import func Darwin.C.tanhf
// @_exported import func Darwin.C.log
// @_exported import func Darwin.C.logf
// @_exported import func Darwin.C.exp
// @_exported import func Darwin.C.expf
// @_exported import func Darwin.C.pow
// @_exported import func Darwin.C.powf
// #else
// @_exported import func Glibc.sin
// @_exported import func Glibc.cos
// @_exported import func Glibc.tan
// @_exported import func Glibc.sinf
// @_exported import func Glibc.cosf
// @_exported import func Glibc.tanf
// @_exported import func Glibc.sinh
// @_exported import func Glibc.cosh
// @_exported import func Glibc.tanh
// @_exported import func Glibc.sinhf
// @_exported import func Glibc.coshf
// @_exported import func Glibc.tanhf
// @_exported import func Glibc.log
// @_exported import func Glibc.logf
// @_exported import func Glibc.exp
// @_exported import func Glibc.expf
// @_exported import func Glibc.pow
// @_exported import func Glibc.powf
// #endif

public extension Tensor where Scalar: SignedNumeric {
    /// Returns the negation of the specified tensor element-wise.
    @inlinable
    @differentiable(vjp: _vjpNegate(_:) where Scalar: TensorFlowFloatingPoint)
    static prefix func - (rhs: Tensor) -> Tensor {
        return _Raw.neg(rhs)
    }
}

internal extension Tensor where Scalar: TensorFlowFloatingPoint {
    @inlinable
    static func _vjpNegate(_ x: Tensor) -> (Tensor, (Tensor) -> Tensor) {
        return (-x, { v in -v })
    }
}

/// Returns the absolute value of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpAbs(_:) where T: TensorFlowFloatingPoint)
public func abs<T: SignedNumeric>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.abs(x)
}

@inlinable
internal func _vjpAbs<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let sign = _Raw.sign(x)
    return (abs(x), { v in v * sign })
}

/// Returns the natural logarithm of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpLog(_:))
public func log<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.log(x)
}

@inlinable
internal func _vjpLog<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    (log(x), { v in v / x })
}

/// Returns the base-two logarithm of the specified tensor element-wise.
@inlinable
@differentiable
public func log2<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    log(x) / T.log(2)
}

/// Returns the base-ten logarithm of the specified tensor element-wise.
@inlinable
@differentiable
public func log10<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    log(x) / T.log(10)
}

/// Returns the logarithm of `1 + x` element-wise.
@inlinable
@differentiable(vjp: _vjpLog1p)
public func log1p<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.log1p(x)
}

@inlinable
func _vjpLog1p<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    (log1p(x), { v in _Raw.xdivy(v, 1 + x) })
}

/// Returns `log(1 - exp(x))` using a numerically stable approach.
///
/// - Note: The approach is shown in Equation 7 of:
///   https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf.
@inlinable
@differentiable
public func log1mexp<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    let isTooSmall = withoutDerivative(at: x) { x in -x .< T(log(2.0)) }
    // This `replacing` will ultimately be a no-op because we will not select this code-path 
    // whenever we use the surrogate `-Tensor(onesLike: x)`.
    let ones = withoutDerivative(at: x) { x in Tensor(onesLike: x) }
    let xSafe = x.replacing(with: -ones, where: isTooSmall)
    return log1p(-exp(xSafe)).replacing(with: log(-expm1(x)), where: isTooSmall)
}

/// Returns the sine of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpSin(_:))
public func sin<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.sin(x)
}

@inlinable
internal func _vjpSin<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    (sin(x), { v in v * cos(x) })
}

/// Returns the cosine of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpCos(_:))
public func cos<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.cos(x)
}

@inlinable
internal func _vjpCos<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    (cos(x), { v in -v * sin(x) })
}

/// Returns the tangent of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpTan(_:))
public func tan<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.tan(x)
}

@inlinable
internal func _vjpTan<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let value = tan(x)
    return (value, { v in v * (1 + value.squared()) })
}

/// Returns the hyperbolic sine of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpSinh(_:))
public func sinh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.sinh(x)
}

@inlinable
internal func _vjpSinh<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    (sinh(x), { v in v * cosh(x) })
}

/// Returns the hyperbolic cosine of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpCosh(_:))
public func cosh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.cosh(x)
}

@inlinable
internal func _vjpCosh<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    (cosh(x), { v in v * sinh(x) })
}

/// Returns the hyperbolic tangent of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpTanh(_:))
public func tanh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.tanh(x)
}

@inlinable
internal func _vjpTanh<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let value = tanh(x)
    return (value, { v in v * (1 - value.squared()) })
}

/// Returns the inverse cosine of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpAcos(_:))
public func acos<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.acos(x)
}

@inlinable
internal func _vjpAcos<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    (acos(x), { v in -v / sqrt(1 - x.squared()) })
}

/// Returns the inverse sine of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpAsin(_:))
public func asin<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.asin(x)
}

@inlinable
internal func _vjpAsin<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    (asin(x), { v in v / sqrt(1 - x.squared()) })
}

/// Returns the inverse tangent of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpAtan(_:))
public func atan<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.atan(x)
}

@inlinable
internal func _vjpAtan<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    (atan(x), { v in v / (1 + x.squared()) })
}

/// Returns the inverse hyperbolic cosine of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpAcosh(_:))
public func acosh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.acosh(x)
}

@inlinable
internal func _vjpAcosh<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    (acosh(x), { v in v / asinh(x) })
}

/// Returns the inverse hyperbolic sine of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpAsinh(_:))
public func asinh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.asinh(x)
}

@inlinable
internal func _vjpAsinh<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    (asinh(x), { v in v / acosh(x) })
}

/// Returns the inverse hyperbolic tangent of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpAtanh(_:))
public func atanh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.atanh(x)
}

@inlinable
internal func _vjpAtanh<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    (atanh(x), { v in v / (1 - x.squared()) })
}

/// Returns the square of the tensor.
public extension Tensor where Scalar: Numeric {
    @inlinable
    @differentiable(wrt: self, vjp: _vjpSquared() where Scalar: TensorFlowFloatingPoint)
    func squared() -> Tensor {
        _Raw.square(self)
    }
}

internal extension Tensor where Scalar: TensorFlowFloatingPoint {
    @inlinable
    func _vjpSquared() -> (Tensor, (Tensor) -> Tensor) {
        (squared(), { 2 * self * $0 })
    }
}

/// Returns the square root of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpSqrt(_:))
public func sqrt<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.sqrt(x)
}

@inlinable
internal func _vjpSqrt<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let value = sqrt(x)
    return (value, { v in v / (2 * value) })
}

/// Returns the inverse square root of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpRsqrt(_:))
public func rsqrt<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.rsqrt(x)
}

@inlinable
internal func _vjpRsqrt<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let value = rsqrt(x)
    return (value, { v in _Raw.rsqrtGrad(value, dy: v) })
}

/// Returns the exponential of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpExp(_:))
public func exp<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.exp(x)
}

@inlinable
internal func _vjpExp<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let value = exp(x)
    return (value, { v in value * v })
}

/// Returns two raised to the power of the specified tensor element-wise.
@inlinable
@differentiable
public func exp2<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    pow(2, x)
}

/// Returns ten raised to the power of the specified tensor element-wise.
@inlinable
@differentiable
public func exp10<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    pow(10, x)
}

/// Returns the exponential of `x - 1` element-wise.
@inlinable
@differentiable(vjp: _vjpExpm1)
public func expm1<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.expm1(x)
}

@inlinable
internal func _vjpExpm1<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let y = expm1(x)
    return (y, { v in v * y })
}

/// Returns the values of the specified tensor rounded to the nearest integer, element-wise.
@inlinable
@differentiable(vjp: _vjpRound)
public func round<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.round(x)
}

@inlinable
internal func _vjpRound<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    (round(x), { v in Tensor<T>(zerosLike: v) })
}

/// Returns the ceiling of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpCeil(_:))
public func ceil<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.ceil(x)
}

@inlinable
internal func _vjpCeil<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    (ceil(x), { _ in Tensor(0).broadcasted(like: x) })
}

/// Returns the floor of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpFloor(_:))
public func floor<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.floor(x)
}

@inlinable
internal func _vjpFloor<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    (floor(x), { _ in Tensor(0).broadcasted(like: x) })
}

/// Returns an indication of the sign of the specified tensor element-wise.
/// Specifically, computes `y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.
@inlinable
@differentiable(vjp: _vjpSign(_:) where T: TensorFlowFloatingPoint)
public func sign<T: Numeric>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.sign(x)
}

@inlinable
internal func _vjpSign<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    (sign(x), { v in Tensor<T>(zerosLike: x) })
}

/// Returns the sigmoid of the specified tensor element-wise.
/// Specifically, computes `1 / (1 + exp(-x))`.
@inlinable
@differentiable(vjp: _vjpSigmoid)
public func sigmoid<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.sigmoid(x)
}

@inlinable
internal func _vjpSigmoid<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let sigmoidValue = sigmoid(x)
    return (sigmoidValue, { v in _Raw.sigmoidGrad(sigmoidValue, dy: v) })
}

/// Returns the log-sigmoid of the specified tensor element-wise. Specifically,
/// `log(1 / (1 + exp(-x)))`. For numerical stability, we use `-softplus(-x)`.
@inlinable
@differentiable
public func logSigmoid<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    -softplus(-x)
}

/// Returns the softplus of the specified tensor element-wise.
/// Specifically, computes `log(exp(features) + 1)`.
@inlinable
@differentiable(vjp: _vjpSoftplus)
public func softplus<T: TensorFlowFloatingPoint>(_ features: Tensor<T>) -> Tensor<T> {
    _Raw.softplus(features: features)
}

@inlinable
internal func _vjpSoftplus<T: TensorFlowFloatingPoint>(
    _ features: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    (softplus(features), { v in _Raw.softplusGrad(gradients: v, features: features)})
}

/// Returns the softsign of the specified tensor element-wise.
/// Specifically, computes `features/ (abs(features) + 1)`.
@inlinable
@differentiable(vjp: _vjpSoftsign)
public func softsign<T: TensorFlowFloatingPoint>(_ features: Tensor<T>) -> Tensor<T> {
    _Raw.softsign(features: features)
}

@inlinable
internal func _vjpSoftsign<T: TensorFlowFloatingPoint>(
    _ features: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    (softsign(features), { v in _Raw.softsignGrad(gradients: v, features: features)})
}

/// Returns the softmax of the specified tensor along the last axis.
/// Specifically, computes `exp(x) / exp(x).sum(alongAxes: -1)`.
@inlinable
@differentiable(vjp: _vjpSoftmax(_:))
public func softmax<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.softmax(logits: x)
}

/// Returns the softmax of the specified tensor along the specified axis.
/// Specifically, computes `exp(x) / exp(x).sum(alongAxes: axis)`.
@inlinable
@differentiable
public func softmax<T: TensorFlowFloatingPoint>(_ x: Tensor<T>, alongAxis axis: Int) -> Tensor<T> {
    let xExp = exp(x)
    return xExp / xExp.sum(alongAxes: Tensor<Int32>(Int32(axis)))
}

@inlinable
func _vjpSoftmax<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let value = softmax(x)
    return (value, { v in
        let sumChannels = (v * value).sum(alongAxes: -1)
        return (v - sumChannels) * value
    })
}

/// Returns the log-softmax of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpLogSoftmax(_:))
public func logSoftmax<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.logSoftmax(logits: x)
}

@inlinable
func _vjpLogSoftmax<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let value = logSoftmax(x)
    return (value, { v in v - v.sum(alongAxes: -1) * exp(value) })
}

/// Returns a tensor by applying an exponential linear unit.
/// Specifically, computes `exp(x) - 1` if < 0, `x` otherwise.
/// See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
/// ](http://arxiv.org/abs/1511.07289)
@inlinable
@differentiable(vjp: _vjpElu)
public func elu<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.elu(features: x)
}

@inlinable
func _vjpElu<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let y = elu(x)
    return (y, { v in _Raw.eluGrad(gradients: v, outputs: y) })
}

/// Returns the Gaussian Error Linear Unit (GELU) activations of the specified tensor element-wise.
///
/// Specifically, `gelu` approximates `xP(X <= x)`, where `P(X <= x)` is the Standard Gaussian
/// cumulative distribution, by computing: x * [0.5 * (1 + tanh[√(2/π) * (x + 0.044715 * x^3)])].
///
/// See [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415).
@inlinable
@differentiable
public func gelu<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    let ratio = Tensor<T>(0.7978845608) // An approximation of √(2/π).
    // An approximation of the Gauss error function.
    // NOTE: This is needed because the compiler otherwise gives an "unable to type-check this
    // in reasonable time" error when the below expressions are written on a single line.
    let approximateErf = tanh(ratio * (x + 0.044715 * pow(x, 3)))
    let cdf = 0.5 * (1.0 + approximateErf)
    return x * cdf
}

/// Returns a tensor by applying the ReLU activation function to the specified tensor element-wise.
/// Specifically, computes `max(0, x)`.
@inlinable
@differentiable(vjp: _vjpRelu(_:))
public func relu<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.relu(features: x)
}

@inlinable
func _vjpRelu<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    (relu(x), { v in _Raw.reluGrad(gradients: v, features: x) })
}

/// Returns a tensor by applying the ReLU6 activation function, namely `min(max(0, x), 6)`.
@inlinable
@differentiable(vjp: _vjpRelu6(_:))
public func relu6<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.relu6(features: x)
}

@inlinable
func _vjpRelu6<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    (relu6(x), { v in _Raw.relu6Grad(gradients: v, features: x)})
}

/// Returns a tensor by applying the leaky ReLU activation function
/// to the specified tensor element-wise.
/// Specifically, computes `max(x, x * alpha)`.
@inlinable
@differentiable(wrt: x, vjp: _vjpLeakyRelu)
public func leakyRelu<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>,
    alpha: Double = 0.2
) -> Tensor<T> {
    _Raw.leakyRelu(features: x, alpha: alpha)
}

@inlinable
func _vjpLeakyRelu<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>,
    alpha: Double
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    (leakyRelu(x, alpha: alpha), { v in
        _Raw.leakyReluGrad(gradients: v, features: x, alpha: alpha)
    })
}

/// Returns a tensor by applying the SeLU activation function, namely
/// `scale * alpha * (exp(x) - 1)` if `x < 0`, and `scale * x` otherwise.
///
/// - Note: This is designed to be used together with the variance scaling layer initializers.
///   Please refer to [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515) for more
///   information.
@inlinable
@differentiable(vjp: _vjpSelu(_:))
public func selu<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.selu(features: x)
}

@inlinable
func _vjpSelu<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let result = selu(x)
    return (result, { v in
        _Raw.seluGrad(gradients: v, outputs: result)
    })
}

public extension Tensor where Scalar: TensorFlowFloatingPoint {
    /// Returns a boolean tensor indicating which elements of `x` are finite.
    @inlinable var isFinite: Tensor<Bool> { _Raw.isFinite(self) }

    /// Returns a boolean tensor indicating which elements of `x` are infinite.
    @inlinable var isInfinite: Tensor<Bool> { _Raw.isInf(self) }

    /// Returns a boolean tensor indicating which elements of `x` are NaN-valued.
    @inlinable var isNaN: Tensor<Bool> { _Raw.isNan(self) }
}

//===------------------------------------------------------------------------------------------===//
// Element-wise Binary Math Functions
//===------------------------------------------------------------------------------------------===//

/// Returns the power of the first tensor to the second tensor.
@inlinable
@differentiable(vjp: _vjpPow(_:_:))
public func pow<T: TensorFlowFloatingPoint>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    _Raw.pow(lhs, rhs)
}

@inlinable
internal func _vjpPow<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>, _ y: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> (Tensor<T>, Tensor<T>)) {
    let value = pow(x, y)
    return (value, { v in
        let safeX = x.replacing(with: Tensor<T>(onesLike: x), where: x .<= 0)
        let lhsGrad = v * y * pow(x, y - 1)
        let rhsGrad = value * v * log(safeX)
        let (lhsShape, rhsShape) = (x.shapeTensor, y.shapeTensor)
        let (lhsAxes, rhsAxes) = _Raw.broadcastGradientArgs(s0: lhsShape, s1: rhsShape)
        return (lhsGrad.sum(squeezingAxes: lhsAxes).reshaped(toShape: lhsShape),
                rhsGrad.sum(squeezingAxes: rhsAxes).reshaped(toShape: rhsShape))
    })
}

/// Returns the power of the scalar to the tensor, broadcasting the scalar.
@inlinable
@differentiable(wrt: rhs where T: TensorFlowFloatingPoint)
public func pow<T: TensorFlowFloatingPoint>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T> {
    pow(Tensor(lhs), rhs)
}

/// Returns the power of the tensor to the scalar, broadcasting the scalar.
@inlinable
@differentiable(wrt: lhs where T: TensorFlowFloatingPoint)
public func pow<T: TensorFlowFloatingPoint>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T> {
    pow(lhs, Tensor(rhs))
}

/// Returns the power of the tensor to the scalar, broadcasting the scalar.
@inlinable
@differentiable
public func pow<T: TensorFlowFloatingPoint>(_ x: Tensor<T>, _ n: Int) -> Tensor<T> {
    pow(x, Tensor(T(n)))
}

/// Returns the element-wise `n`th root of the tensor.
@inlinable
@differentiable
public func root<T: TensorFlowFloatingPoint>(_ x: Tensor<T>, _ n: Int) -> Tensor<T> {
    sign(x) * pow(abs(x), Tensor(T(1) / T(n)))
}

/// Returns the squared difference between `x` and `y`.
/// - Returns: `(x - y) ^ 2`.
@inlinable
@differentiable(vjp: _vjpSquaredDifference where T: TensorFlowFloatingPoint)
public func squaredDifference<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T> {
    _Raw.squaredDifference(x, y)
}

@inlinable
internal func _vjpSquaredDifference<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> (Tensor<T>, Tensor<T>)) {
    (squaredDifference(x, y), { seed in
        let lhsGrad = 2 * seed * (x - y)
        let rhsGrad = -lhsGrad
        let (lhsShape, rhsShape) = (x.shapeTensor, y.shapeTensor)
        let (lhsAxes, rhsAxes) = _Raw.broadcastGradientArgs(s0: lhsShape, s1: rhsShape)
        return (lhsGrad.sum(squeezingAxes: lhsAxes).reshaped(toShape: lhsShape),
                rhsGrad.sum(squeezingAxes: rhsAxes).reshaped(toShape: rhsShape))
    })
}

/// Returns the element-wise maximum of two tensors.
/// - Note: `max` supports broadcasting.
@inlinable
@differentiable(vjp: _vjpMax(_:_:) where T: TensorFlowFloatingPoint)
public func max<T>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable {
    _Raw.maximum(lhs, rhs)
}

@inlinable
internal func _vjpMax<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> (Tensor<T>, Tensor<T>)) {
    let value = max(x, y)
    return (value, { v in
        _vjpMinMaxHelper(x, y, originalValue: value, seed: v, comparisonOperation: .>=)
    })
}

/// Returns the element-wise maximum of the scalar and the tensor, broadcasting the scalar.
@inlinable
@differentiable(wrt: rhs where T: TensorFlowFloatingPoint)
public func max<T>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable {
    max(Tensor(lhs), rhs)
}

/// Returns the element-wise maximum of the scalar and the tensor, broadcasting the scalar.
@inlinable
@differentiable(wrt: lhs where T: TensorFlowFloatingPoint)
public func max<T>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T> where T: Numeric & Comparable {
    max(lhs, Tensor(rhs))
}

/// Returns the element-wise minimum of two tensors.
/// - Note: `min` supports broadcasting.
@inlinable
@differentiable(vjp: _vjpMin(_:_:) where T: TensorFlowFloatingPoint)
public func min<T>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable {
    _Raw.minimum(lhs, rhs)
}

@inlinable
internal func _vjpMin<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> (Tensor<T>, Tensor<T>)) {
    let value = min(x, y)
    return (value, { v in
        _vjpMinMaxHelper(x, y, originalValue: value, seed: v, comparisonOperation: .<=)
    })
}

/// Returns the element-wise minimum of the scalar and the tensor, broadcasting the scalar.
@inlinable
@differentiable(wrt: rhs where T: TensorFlowFloatingPoint)
public func min<T>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable {
    min(Tensor(lhs), rhs)
}

/// Returns the element-wise minimum of the scalar and the tensor, broadcasting the scalar.
@inlinable
@differentiable(wrt: lhs where T: TensorFlowFloatingPoint)
public func min<T>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T> where T: Numeric & Comparable {
    min(lhs, Tensor(rhs))
}

@inlinable
internal func _vjpMinMaxHelper<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>,
    _ y: Tensor<T>,
    originalValue: Tensor<T>,
    seed: Tensor<T>,
    comparisonOperation: (Tensor<T>, Tensor<T>) -> Tensor<Bool>
) -> (Tensor<T>, Tensor<T>) {
    let mask = Tensor<T>(comparisonOperation(x, y))
    let lhsGrad = seed * mask
    let rhsGrad = seed * (1 - mask)
    let (lhsShape, rhsShape) = (x.shapeTensor, y.shapeTensor)
    let (lhsAxes, rhsAxes) = _Raw.broadcastGradientArgs(s0: lhsShape, s1: rhsShape)
    return (lhsGrad.sum(squeezingAxes: lhsAxes).reshaped(toShape: lhsShape),
            rhsGrad.sum(squeezingAxes: rhsAxes).reshaped(toShape: rhsShape))
}

/// Returns the cosine similarity between `x` and `y`.
@differentiable
public func cosineSimilarity<Scalar: TensorFlowFloatingPoint>(
    _ x: Tensor<Scalar>,
    _ y: Tensor<Scalar>
) -> Tensor<Scalar> {
    (x * y).sum() / (sqrt(x.squared().sum()) * sqrt(y.squared().sum()))
}

/// Returns the cosine distance between `x` and `y`. Cosine distance is defined as
/// `1 - cosineSimilarity(x, y)`.
@differentiable
public func cosineDistance<Scalar: TensorFlowFloatingPoint>(
    _ x: Tensor<Scalar>,
    _ y: Tensor<Scalar>
) -> Tensor<Scalar> {
    1 - cosineSimilarity(x, y)
}

//===------------------------------------------------------------------------------------------===//
// Selection Functions
//===------------------------------------------------------------------------------------------===//

public extension Tensor {
    /// Replaces elements of this tensor with `other` in the lanes where `mask` is
    /// `true`.
    ///
    /// - Precondition: `self` and `other` must have the same shape. If
    ///   `self` and `other` are scalar, then `mask` must also be scalar. If
    ///   `self` and `other` have rank greater than or equal to `1`, then `mask`
    ///   must be either have the same shape as `self` or be a 1-D `Tensor` such
    ///   that `mask.scalarCount == self.shape[0]`.
    @inlinable
    @differentiable(wrt: (self, other), vjp: _vjpReplacing where Scalar: TensorFlowFloatingPoint)
    func replacing(with other: Tensor, where mask: Tensor<Bool>) -> Tensor {
        _Raw.select(condition: mask, t: other, e: self)
    }
}

internal extension Tensor where Scalar: TensorFlowFloatingPoint {
    @inlinable
    func _vjpReplacing(
        with other: Tensor,
        where mask: Tensor<Bool>
    ) -> (Tensor, (Tensor) -> (Tensor, Tensor)) {
        return (replacing(with: other, where: mask), { v in
            let zeros = Tensor(zeros: v.shape)
            return (v.replacing(with: zeros, where: mask), zeros.replacing(with: v, where: mask))
        })
    }
}

//===------------------------------------------------------------------------------------------===//
// Reduction Functions
//===------------------------------------------------------------------------------------------===//

public extension Tensor where Scalar == Bool {
    /// Returns `true` if all scalars are equal to `true`. Otherwise, returns `false`.
    // NOTE: This overload is necessary, otherwise `all()` would refer to the variadic method
    // `all(squeezingAxes:)` with zero indices.
    @inlinable
    func all() -> Bool {
        let axes = Tensor<Int32>(rangeFrom: 0, to: Int32(rank), stride: 1)
        return _Raw.all(self, reductionIndices: axes).scalarized()
    }

    /// Returns `true` if any scalars are equal to `true`. Otherwise, returns `false`.
    // NOTE: This overload is necessary, otherwise `any()` would refer to the variadic method
    // `any(squeezingAxes:)` with zero indices.
    @inlinable
    func any() -> Bool {
        let axes = Tensor<Int32>(rangeFrom: 0, to: Int32(rank), stride: 1)
        return _Raw.any(self, reductionIndices: axes).scalarized()
    }

    /// Performs a logical AND operation along the specified axes. The reduced dimensions are
    /// removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    func all(squeezingAxes axes: Int...) -> Tensor {
        let axes = axes.map(Int32.init)
        return _Raw.all(self, reductionIndices: Tensor<Int32>(axes), keepDims: false)
    }

    /// Performs a logical AND operation along the specified axes. The reduced dimensions are
    /// removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    func any(squeezingAxes axes: Int...) -> Tensor {
        let axes = axes.map(Int32.init)
        return _Raw.any(self, reductionIndices: Tensor<Int32>(axes), keepDims: false)
    }

    /// Performs a logical AND operation along the specified axes. The reduced dimensions are
    /// retained with value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    func all(alongAxes axes: Int...) -> Tensor {
        let axes = axes.map(Int32.init)
        return _Raw.all(self, reductionIndices: Tensor<Int32>(axes), keepDims: true)
    }

    /// Performs a logical OR operation along the specified axes. The reduced
    /// dimensions are retained with value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    func any(alongAxes axes: Int...) -> Tensor {
        let axes = axes.map(Int32.init)
        return _Raw.any(self, reductionIndices: Tensor<Int32>(axes), keepDims: true)
    }
}

public extension Tensor where Scalar: Numeric & Comparable {
    // NOTE: This overload is necessary, otherwise `min()` would refer to the variadic method
    // `min(squeezingAxes:)` with zero indices.
    @inlinable
    @differentiable(where Scalar: TensorFlowFloatingPoint)
    func min() -> Tensor {
        let axes = Tensor<Int32>(rangeFrom: 0, to: Int32(rank), stride: 1)
        return min(squeezingAxes: axes)
    }

    // NOTE: This overload is necessary, otherwise `max()` would refer to the variadic method
    // `max(squeezingAxes:)` with zero indices.
    @inlinable
    @differentiable(where Scalar: TensorFlowFloatingPoint)
    func max() -> Tensor {
        let axes = Tensor<Int32>(rangeFrom: 0, to: Int32(rank), stride: 1)
        return max(squeezingAxes: axes)
    }

    /// Returns the maximum values along the specified axes. The reduced dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(
        wrt: self,
        vjp: _vjpMinOrMax(squeezingAxes:) where Scalar: TensorFlowFloatingPoint)
    func max(squeezingAxes axes: Tensor<Int32>) -> Tensor {
        return _Raw.max(self, reductionIndices: axes, keepDims: false)
    }

    /// Returns the maximum values along the specified axes. The reduced dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func max(squeezingAxes axes: [Int]) -> Tensor {
        // TODO(TF-433): Remove workaround for differentiating `map`.
        let axes = {axes.map(Int32.init)}()
        return max(squeezingAxes: Tensor<Int32>(axes))
    }

    /// Returns the maximum values along the specified axes. The reduced dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func max(squeezingAxes axes: Int...) -> Tensor {
        max(squeezingAxes: axes)
    }

    /// Returns the minimum values along the specified axes. The reduced dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(
        wrt: self,
        vjp: _vjpMinOrMax(squeezingAxes:) where Scalar: TensorFlowFloatingPoint)
    func min(squeezingAxes axes: Tensor<Int32>) -> Tensor {
        _Raw.min(self, reductionIndices: axes, keepDims: false)
    }

    /// Returns the minimum values along the specified axes. The reduced dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func min(squeezingAxes axes: [Int]) -> Tensor {
        // TODO(TF-433): Remove workaround for differentiating `map`.
        let axes = {axes.map(Int32.init)}()
        return min(squeezingAxes: Tensor<Int32>(axes))
    }

    /// Returns the minimum values along the specified axes. The reduced dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func min(squeezingAxes axes: Int...) -> Tensor {
        min(squeezingAxes: axes)
    }

    /// Returns the indices of the maximum values along the specified axes. The reduced dimensions
    /// are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    func argmax(squeezingAxis axis: Int) -> Tensor<Int32> {
        _Raw.argMax(self, dimension: Tensor<Int32>(Int32(axis)))
    }

    /// Returns the indices of the minimum values along the specified axes. The reduced dimensions
    /// are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    func argmin(squeezingAxis axis: Int) -> Tensor<Int32> {
        _Raw.argMin(self, dimension: Tensor<Int32>(Int32(axis)))
    }

    /// Returns the minimum along the specified axes. The reduced dimensions are retained with
    /// value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpMinOrMax(alongAxes:) where Scalar: TensorFlowFloatingPoint)
    func min(alongAxes axes: Tensor<Int32>) -> Tensor {
        _Raw.min(self, reductionIndices: axes, keepDims: true)
    }

    /// Returns the minimum along the specified axes. The reduced dimensions are retained with
    /// value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func min(alongAxes axes: [Int]) -> Tensor {
        // TODO(TF-433): Remove workaround for differentiating `map`.
        let axes = {axes.map(Int32.init)}()
        return min(alongAxes: Tensor<Int32>(axes))
    }

    /// Returns the minimum along the specified axes. The reduced dimensions are retained with
    /// value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func min(alongAxes axes: Int...) -> Tensor {
        min(alongAxes: axes)
    }

    /// Returns the minimum along the specified axes. The reduced dimensions are retained with
    /// value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpMinOrMax(alongAxes:) where Scalar: TensorFlowFloatingPoint)
    func max(alongAxes axes: Tensor<Int32>) -> Tensor {
        _Raw.max(self, reductionIndices: axes, keepDims: true)
    }

    /// Returns the minimum along the specified axes. The reduced dimensions are retained with
    /// value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func max(alongAxes axes: [Int]) -> Tensor {
        // TODO(TF-433): Remove workaround for differentiating `map`.
        let axes = {axes.map(Int32.init)}()
        return max(alongAxes: Tensor<Int32>(axes))
    }

    /// Returns the minimum along the specified axes. The reduced dimensions are retained with
    /// value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func max(alongAxes axes: Int...) -> Tensor {
        max(alongAxes: axes)
    }

    /// Returns the index of the maximum value of the flattened scalars.
    @inlinable
    func argmax() -> Tensor<Int32> {
        flattened().argmax(squeezingAxis: 0)
    }

    /// Returns the index of the minimum value of the flattened scalars.
    @inlinable
    func argmin() -> Tensor<Int32> {
        flattened().argmin(squeezingAxis: 0)
    }
}

internal extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  func _vjpMinOrMax(squeezingAxes axes: Tensor<Int32>) -> (Tensor, (Tensor) -> Tensor) {
    let result = max(squeezingAxes: axes)
    return (result, { v in
      let yUnsqueezed = result.expandingShape(at: axes.scalars.map { Int($0) })
      let gradientUnsqueezed = v.expandingShape(at: axes.scalars.map { Int($0) })

      // Compute the number of selected (maximum or minimum) elements in each reduction dimension.
      // If there are multiple minimum or maximum elements then the gradient will be divided between
      // them.
      let indicators = Tensor(yUnsqueezed .== self)
      let selectedCount = indicators.sum(alongAxes: axes)

      return gradientUnsqueezed.broadcasted(toShape: self.shapeTensor) * indicators / selectedCount
    })
  }

  @inlinable
  func _vjpMinOrMax(alongAxes axes: Tensor<Int32>) -> (Tensor, (Tensor) -> Tensor) {
    let result = max(alongAxes: axes)
    return (result, { v in
      // Compute the number of selected (maximum or minimum) elements in each reduction dimension.
      // If there are multiple minimum or maximum elements then the gradient will be divided between
      // them.
      let indicators = Tensor(result .== self)
      let selectedCount = indicators.sum(alongAxes: axes)
      return v.broadcasted(toShape: self.shapeTensor) * indicators / selectedCount
    })
  }
}

// MARK: - Numeric Reductions

public extension Tensor where Scalar: Numeric {
    // MARK: - Sum

    /// Returns the sum along the specified axes. The reduced dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpSum(squeezingAxes:) where Scalar: TensorFlowFloatingPoint)
    func sum(squeezingAxes axes: Tensor<Int32>) -> Tensor {
        _Raw.sum(self, reductionIndices: axes, keepDims: false)
    }

    /// Returns the sum along the specified axes. The reduced dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func sum(squeezingAxes axes: [Int]) -> Tensor {
        // TODO(TF-433): Remove workaround for differentiating `map`.
        let axes = {axes.map(Int32.init)}()
        return sum(squeezingAxes: Tensor<Int32>(axes))
    }

    /// Returns the sum along the specified axes. The reduced dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func sum(squeezingAxes axes: Int...) -> Tensor {
        sum(squeezingAxes: axes)
    }

    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func sum() -> Tensor {
        flattened().sum(squeezingAxes: 0)
    }

    /// Returns the sum along the specified axes. The reduced dimensions are retained with value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpSum(alongAxes:) where Scalar: TensorFlowFloatingPoint)
    func sum(alongAxes axes: Tensor<Int32>) -> Tensor {
        _Raw.sum(self, reductionIndices: axes, keepDims: true)
    }

    /// Returns the sum along the specified axes. The reduced dimensions are retained with value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func sum(alongAxes axes: [Int]) -> Tensor {
        // TODO(TF-433): Remove workaround for differentiating `map`.
        let axes = {axes.map(Int32.init)}()
        return sum(alongAxes: Tensor<Int32>(axes))
    }

    /// Returns the sum along the specified axes. The reduced dimensions are retained with value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func sum(alongAxes axes: Int...) -> Tensor {
        sum(alongAxes: axes)
    }

    // MARK: - Product

    /// Returns the product along the specified axes. The reduced dimensions are removed.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
    // TODO: Make this @differentiable.
    @inlinable
    func product(squeezingAxes axes: Tensor<Int32>) -> Tensor {
        _Raw.prod(self, reductionIndices: axes, keepDims: false)
    }

    /// Returns the product along the specified axes. The reduced dimensions are removed.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
    @inlinable
    func product(squeezingAxes axes: [Int]) -> Tensor {
        // TODO(TF-433): Remove workaround for differentiating `map`.
        let axes = {axes.map(Int32.init)}()
        return product(squeezingAxes: Tensor<Int32>(axes))
    }

    /// Returns the product along the specified axes. The reduced dimensions are removed.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
    @inlinable
    func product(squeezingAxes axes: Int...) -> Tensor {
        product(squeezingAxes: axes)
    }

    @inlinable
    func product() -> Tensor {
        flattened().product(squeezingAxes: 0)
    }

    /// Returns the product along the specified axes. The reduced dimensions are retained with
    /// value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    func product(alongAxes axes: Tensor<Int32>) -> Tensor {
        _Raw.prod(self, reductionIndices: axes, keepDims: true)
    }

    /// Returns the product along the specified axes. The reduced dimensions are retained with
    /// value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    func product(alongAxes axes: [Int]) -> Tensor {
        // TODO(TF-433): Remove workaround for differentiating `map`.
        let axes = {axes.map(Int32.init)}()
        return product(alongAxes: Tensor<Int32>(axes))
    }

    /// Returns the product along the specified axes. The reduced dimensions are retained with
    /// value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    func product(alongAxes axes: Int...) -> Tensor {
        product(alongAxes: axes)
    }

    // MARK: - Mean

    /// Returns the arithmetic mean along the specified axes. The reduced dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpMean(squeezingAxes:) where Scalar: TensorFlowFloatingPoint)
    func mean(squeezingAxes axes: Tensor<Int32>) -> Tensor {
        _Raw.mean(self, reductionIndices: axes, keepDims: false)
    }

    /// Returns the arithmetic mean along the specified axes. The reduced dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func mean(squeezingAxes axes: [Int]) -> Tensor {
        // TODO(TF-433): Remove workaround for differentiating `map`.
        let axes = {axes.map(Int32.init)}()
        return mean(squeezingAxes: Tensor<Int32>(axes))
    }

    /// Returns the arithmetic mean along the specified axes. The reduced dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func mean(squeezingAxes axes: Int...) -> Tensor {
        mean(squeezingAxes: axes)
    }

    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func mean() -> Tensor {
        flattened().mean(squeezingAxes: [0])
    }

    /// Returns the arithmetic mean along the specified axes. The reduced dimensions are retained
    /// with value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpMean(alongAxes:) where Scalar: TensorFlowFloatingPoint)
    func mean(alongAxes axes: Tensor<Int32>) -> Tensor {
        _Raw.mean(self, reductionIndices: axes, keepDims: true)
    }

    /// Returns the arithmetic mean along the specified axes. The reduced dimensions are retained
    /// with value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func mean(alongAxes axes: [Int]) -> Tensor {
        // TODO(TF-433): Remove workaround for differentiating `map`.
        let axes = {axes.map(Int32.init)}()
        return mean(alongAxes: Tensor<Int32>(axes))
    }

    /// Returns the arithmetic mean along the specified axes. The reduced dimensions are retained
    /// with value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func mean(alongAxes axes: Int...) -> Tensor {
        mean(alongAxes: axes)
    }

    // MARK: - Variance

    /// Returns the variance along the specified axes. The reduced dimensions are removed. Does not
    /// apply Bessel's correction.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func variance(squeezingAxes axes: Tensor<Int32>) -> Tensor {
        let squaredDiff = (self - mean(alongAxes: axes)).squared()
        return squaredDiff.mean(squeezingAxes: axes)
    }

    /// Returns the variance along the specified axes. The reduced dimensions are removed. Does not
    /// apply Bessel's correction.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func variance(squeezingAxes axes: [Int]) -> Tensor {
        // TODO(TF-433): Remove workaround for differentiating `map`.
        let axes = {axes.map(Int32.init)}()
        return variance(squeezingAxes: Tensor<Int32>(axes))
    }

    /// Returns the variance along the specified axes. The reduced dimensions are retained with
    /// value 1. Does not apply Bessel's correction.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func variance(squeezingAxes axes: Int...) -> Tensor {
        variance(squeezingAxes: axes)
    }

    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func variance() -> Tensor {
        let mean = self.mean()
        let squaredDiff = (self - mean).squared()
        return squaredDiff.mean()
    }

    /// Returns the variance along the specified axes. The reduced dimensions are retained with
    /// value 1. Does not apply Bessel's correction.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func variance(alongAxes axes: Tensor<Int32>) -> Tensor {
        let squaredDiff = (self - mean(alongAxes: axes)).squared()
        return squaredDiff.mean(alongAxes: axes)
    }

    /// Returns the variance along the specified axes. The reduced dimensions are retained with
    /// value 1. Does not apply Bessel's correction.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func variance(alongAxes axes: [Int]) -> Tensor {
        // TODO(TF-433): Remove workaround for differentiating `map`.
        let axes = {axes.map(Int32.init)}()
        return variance(alongAxes: Tensor<Int32>(axes))
    }

    /// Returns the variance along the specified axes. The reduced dimensions are retained with
    /// value 1. Does not apply Bessel's correction.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func variance(alongAxes axes: Int...) -> Tensor {
        variance(alongAxes: axes)
    }

    /// Returns the cumulative sum of this tensor along the specified axis. By default, this
    /// function performs an inclusive cumulative sum which means that the first element of the
    /// input is identical to the first element of the output:
    /// ```
    /// Tensor<Float>([a, b, c]).cumulativeSum() = Tensor<Float>([a, a + b, a + b + c])
    /// ```
    /// By setting the `exclusive` argument to `true`, an exclusive cumulative sum is performed
    /// instead:
    /// ```
    /// Tensor<Float>([a, b, c]).cumulativeSum(exclusive: true) = Tensor<Float>([0, a, a + b])
    /// ```
    /// By setting the `reverse` argument to `true`, the cumulative sum is performed in the
    /// opposite direction:
    /// ```
    /// Tensor<Float>([a, b, c]).cumulativeSum(reverse: true) ==
    ///     Tensor<Float>([a + b + c, a + b, a])
    /// ```
    /// This is more efficient than separately reversing the resulting tensor.
    ///
    /// - Parameters:
    ///   - axis: Axis along which to perform the cumulative sum operation.
    ///   - exclusive: Indicates whether to perform an exclusive cumulative sum.
    ///   - reverse: Indicates whether to perform the cumulative sum in reversed order.
    /// - Returns: Result of the cumulative sum operation.
    /// - Precondition: `axis` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func cumulativeSum(
        alongAxis axis: Int,
        exclusive: Bool = false,
        reverse: Bool = false
    ) -> Tensor {
        cumulativeSum(
            alongAxis: Tensor<Int32>(Int32(axis)),
            exclusive: exclusive,
            reverse: reverse)
    }

    /// Returns the cumulative sum of this tensor along the specified axis. By default, this
    /// function performs an inclusive cumulative sum which means that the first element of the
    /// input is identical to the first element of the output:
    /// ```
    /// Tensor<Float>([a, b, c]).cumulativeSum() = Tensor<Float>([a, a + b, a + b + c])
    /// ```
    /// By setting the `exclusive` argument to `true`, an exclusive cumulative sum is performed
    /// instead:
    /// ```
    /// Tensor<Float>([a, b, c]).cumulativeSum(exclusive: true) = Tensor<Float>([0, a, a + b])
    /// ```
    /// By setting the `reverse` argument to `true`, the cumulative sum is performed in the
    /// opposite direction:
    /// ```
    /// Tensor<Float>([a, b, c]).cumulativeSum(reverse: true) ==
    ///     Tensor<Float>([a + b + c, a + b, a])
    /// ```
    /// This is more efficient than separately reversing the resulting tensor.
    ///
    /// - Parameters:
    ///   - axis: Axis along which to perform the cumulative sum operation.
    ///   - exclusive: Indicates whether to perform an exclusive cumulative sum.
    ///   - reverse: Indicates whether to perform the cumulative sum in reversed order.
    /// - Returns: Result of the cumulative sum operation.
    /// - Precondition: `axis.rank` must be `0`.
    /// - Precondition: `axis` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpCumulativeSum where Scalar: TensorFlowFloatingPoint)
    func cumulativeSum(
        alongAxis axis: Tensor<Int32>,
        exclusive: Bool = false,
        reverse: Bool = false
    ) -> Tensor {
        _Raw.cumsum(self, axis: axis, exclusive: exclusive, reverse: reverse)
    }

    /// Returns the cumulative product of this tensor along the specified axis. By default, this
    /// function performs an inclusive cumulative product which means that the first element of the
    /// input is identical to the first element of the output:
    /// ```
    /// Tensor<Float>([a, b, c]).cumulativeProduct() = Tensor<Float>([a, a * b, a * b * c])
    /// ```
    /// By setting the `exclusive` argument to `true`, an exclusive cumulative product is performed
    /// instead:
    /// ```
    /// Tensor<Float>([a, b, c]).cumulativeProduct(exclusive: true) = Tensor<Float>([1, a, a * b])
    /// ```
    /// By setting the `reverse` argument to `true`, the cumulative product is performed in the
    /// opposite direction:
    /// ```
    /// Tensor<Float>([a, b, c]).cumulativeProduct(reverse: true) ==
    ///     Tensor<Float>([a * b * c, a * b, a])
    /// ```
    /// This is more efficient than separately reversing the resulting tensor.
    ///
    /// - Parameters:
    ///   - axis: Axis along which to perform the cumulative product operation.
    ///   - exclusive: Indicates whether to perform an exclusive cumulative product.
    ///   - reverse: Indicates whether to perform the cumulative product in reversed order.
    /// - Returns: Result of the cumulative product operation.
    /// - Precondition: `axis` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func cumulativeProduct(
        alongAxis axis: Int,
        exclusive: Bool = false,
        reverse: Bool = false
    ) -> Tensor {
        cumulativeProduct(
            alongAxis: Tensor<Int32>(Int32(axis)),
            exclusive: exclusive,
            reverse: reverse)
    }

    /// Returns the cumulative product of this tensor along the specified axis. By default, this
    /// function performs an inclusive cumulative product which means that the first element of the
    /// input is identical to the first element of the output:
    /// ```
    /// Tensor<Float>([a, b, c]).cumulativeProduct() = Tensor<Float>([a, a * b, a * b * c])
    /// ```
    /// By setting the `exclusive` argument to `true`, an exclusive cumulative product is performed
    /// instead:
    /// ```
    /// Tensor<Float>([a, b, c]).cumulativeProduct(exclusive: true) = Tensor<Float>([1, a, a * b])
    /// ```
    /// By setting the `reverse` argument to `true`, the cumulative product is performed in the
    /// opposite direction:
    /// ```
    /// Tensor<Float>([a, b, c]).cumulativeProduct(reverse: true) ==
    ///     Tensor<Float>([a * b * c, a * b, a])
    /// ```
    /// This is more efficient than separately reversing the resulting tensor.
    ///
    /// - Parameters:
    ///   - axis: Axis along which to perform the cumulative product operation.
    ///   - exclusive: Indicates whether to perform an exclusive cumulative product.
    ///   - reverse: Indicates whether to perform the cumulative product in reversed order.
    /// - Returns: Result of the cumulative product operation.
    /// - Precondition: `axis` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpCumulativeProduct where Scalar: TensorFlowFloatingPoint)
    func cumulativeProduct(
        alongAxis axis: Tensor<Int32>,
        exclusive: Bool = false,
        reverse: Bool = false
    ) -> Tensor {
        _Raw.cumprod(self, axis: axis, exclusive: exclusive, reverse: reverse)
    }
}

internal extension Tensor where Scalar: TensorFlowFloatingPoint {
    @inlinable
    func _vjpSum(alongAxes axes: Tensor<Int32>) -> (Tensor, (Tensor) -> Tensor) {
        let value = sum(alongAxes: axes)
        return (value, { [shape = shapeTensor] in $0.broadcasted(toShape: shape) })
    }

    @inlinable
    func _vjpSum(squeezingAxes axes: Tensor<Int32>) -> (Tensor, (Tensor) -> Tensor) {
        let value = sum(squeezingAxes: axes)
        return (value, { [shape = shapeTensor] v in
            let unsqueezed = v.expandingShape(at: axes.scalars.map { Int($0) })
            return unsqueezed.broadcasted(toShape: shape)
        })
    }

    @inlinable
    func _vjpMean(alongAxes axes: Tensor<Int32>) -> (Tensor, (Tensor) -> Tensor) {
        let value = mean(alongAxes: axes)
        let count = _Raw.gather(params: shapeTensor, indices: axes).product()
        return (value, { [shape = shapeTensor] in $0.broadcasted(toShape: shape) / Tensor(count) })
    }

    @inlinable
    func _vjpMean(squeezingAxes axes: Tensor<Int32>) -> (Tensor, (Tensor) -> Tensor) {
        let value = mean(squeezingAxes: axes)
        let count = _Raw.gather(params: shapeTensor, indices: axes).product()
        return (value, { [shape = shapeTensor] v in
            let unsqueezed = v.expandingShape(at: axes.scalars.map { Int($0) })
            return unsqueezed.broadcasted(toShape: shape) / Tensor(count)
        })
    }

    @inlinable
    func _vjpCumulativeSum(
        alongAxis axis: Tensor<Int32>,
        exclusive: Bool = false,
        reverse: Bool = false
    ) -> (Tensor, (Tensor) -> Tensor) {
        (cumulativeSum(alongAxis: axis, exclusive: exclusive, reverse: reverse), { v in
            v.cumulativeSum(alongAxis: axis, exclusive: exclusive, reverse: !reverse)
        })
    }

    @inlinable
    func _vjpCumulativeProduct(
        alongAxis axis: Tensor<Int32>,
        exclusive: Bool = false,
        reverse: Bool = false
    ) -> (Tensor, (Tensor) -> Tensor) {
        let result = cumulativeProduct(alongAxis: axis, exclusive: exclusive, reverse: reverse)
        return (result, { v in
            (result * v).cumulativeSum(
                alongAxis: axis,
                exclusive: exclusive,
                reverse: !reverse
            ) / self
        })
    }
}

// TODO: Consider making the return type be generic over `FloatingPoint` types
// so that `self`'s scalar type can be any `Numeric` type.
public extension Tensor where Scalar: TensorFlowFloatingPoint {
    /// Returns the standard deviation of the elements along the specified axes. The reduced
    /// dimensions are retained with value `1`. Does not apply Bessel's correction.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self)
    func standardDeviation(squeezingAxes axes: Tensor<Int32>) -> Tensor {
        TensorFlow.sqrt(variance(squeezingAxes: axes))
    }

    /// Returns the standard deviation of the elements along the specified axes. The reduced
    /// dimensions are retained with value `1`. Does not apply Bessel's correction.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self)
    func standardDeviation(squeezingAxes axes: [Int]) -> Tensor {
        TensorFlow.sqrt(variance(squeezingAxes: axes))
    }

    /// Returns the standard deviation of the elements along the specified axes. The reduced
    /// dimensions are retained with value `1`. Does not apply Bessel's correction.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self)
    func standardDeviation(squeezingAxes axes: Int...) -> Tensor {
        standardDeviation(squeezingAxes: axes)
    }

    /// Returns the standard deviation of all elements in this tensor. 
    /// Does not apply Bessel's correction.
    ///
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self)
    func standardDeviation() -> Tensor {
        // Reduce along all dimensions.
        standardDeviation(squeezingAxes: Array(0..<shape.rank))
    }

    /// Returns the standard deviation of the elements along the specified axes. The reduced
    /// dimensions are retained with value `1`. Does not apply Bessel's correction.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self)
    func standardDeviation(alongAxes axes: Tensor<Int32>) -> Tensor {
        TensorFlow.sqrt(variance(alongAxes: axes))
    }

    /// Returns the standard deviation of the elements along the specified axes. The reduced
    /// dimensions are retained with value `1`. Does not apply Bessel's correction.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self)
    func standardDeviation(alongAxes axes: [Int]) -> Tensor {
        // TODO(TF-433): Remove workaround for differentiating `map`.
        let axes = {axes.map(Int32.init)}()
        return standardDeviation(alongAxes: Tensor<Int32>(axes))
    }

    /// Returns the standard deviation of the elements along the specified axes. The reduced
    /// dimensions are retained with value `1`. Does not apply Bessel's correction.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self)
    func standardDeviation(alongAxes axes: Int...) -> Tensor {
        TensorFlow.sqrt(variance(alongAxes: axes))
    }

    /// Returns `log(exp(self).sum(squeezingAxes: axes))`. The reduced dimensions are removed.
    /// 
    /// This function is more numerically stable than computing
    /// `log(exp(self).sum(squeezingAxes: axes))` directly. It avoids overflows caused by computing 
    /// the `exp` of large inputs and underflows caused by computing the `log` of small inputs.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self)
    func logSumExp(squeezingAxes axes: Tensor<Int32>) -> Tensor {
        let rawMax = max(alongAxes: axes)
        let offset = withoutDerivative(at: rawMax) { rawMax in 
            rawMax.replacing(
                with: Tensor<Scalar>(zerosLike: rawMax),
                where: rawMax.isFinite)
        }
        let result = TensorFlow.log(TensorFlow.exp(self - offset).sum(squeezingAxes: axes))
        let resultShape = withoutDerivative(at: result.shapeTensor)
        return result + offset.reshaped(toShape: resultShape)
    }

    /// Returns `log(exp(self).sum(squeezingAxes: axes))`. The reduced dimensions are removed.
    /// 
    /// This function is more numerically stable than computing
    /// `log(exp(self).sum(squeezingAxes: axes))` directly. It avoids overflows caused by computing 
    /// the `exp` of large inputs and underflows caused by computing the `log` of small inputs.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self)
    func logSumExp(squeezingAxes axes: [Int]) -> Tensor {
        // TODO(TF-433): Remove workaround for differentiating `map`.
        let axes = withoutDerivative(at: axes) { $0.map(Int32.init) }
        return logSumExp(squeezingAxes: Tensor<Int32>(axes))
    }

    /// Returns `log(exp(self).sum(squeezingAxes: axes))`. The reduced dimensions are removed.
    /// 
    /// This function is more numerically stable than computing
    /// `log(exp(self).sum(squeezingAxes: axes))` directly. It avoids overflows caused by computing 
    /// the `exp` of large inputs and underflows caused by computing the `log` of small inputs.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self)
    func logSumExp(squeezingAxes axes: Int...) -> Tensor {
        logSumExp(squeezingAxes: axes)
    }

    /// Returns `log(exp(self).sum())`. The result is a scalar.
    ///
    /// This function is more numerically stable than computing `log(exp(self).sum())` directly. It
    /// avoids overflows caused by computing the `exp` of large inputs and underflows caused by 
    /// computing the `log` of small inputs.
    @inlinable
    @differentiable(wrt: self)
    func logSumExp() -> Tensor {
        logSumExp(squeezingAxes: Array(0..<shape.rank))
    }

    /// Returns `log(exp(self).sum(alongAxes: axes))`. The reduced dimensions are retained with 
    /// value `1`.
    ///
    /// This function is more numerically stable than computing
    /// `log(exp(self).sum(alongAxes: axes))` directly. It avoids overflows caused by computing 
    /// the `exp` of large inputs and underflows caused by computing the `log` of small inputs.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self)
    func logSumExp(alongAxes axes: Tensor<Int32>) -> Tensor {
        let rawMax = max(alongAxes: axes)
        let offset = withoutDerivative(at: rawMax) { rawMax in 
            rawMax.replacing(
                with: Tensor<Scalar>(zerosLike: rawMax),
                where: rawMax.isFinite)
        }
        let result = TensorFlow.log(TensorFlow.exp(self - offset).sum(alongAxes: axes))
        return result + offset
    }

    /// Returns `log(exp(self).sum(alongAxes: axes))`. The reduced dimensions are retained with 
    /// value `1`.
    ///
    /// This function is more numerically stable than computing
    /// `log(exp(self).sum(alongAxes: axes))` directly. It avoids overflows caused by computing 
    /// the `exp` of large inputs and underflows caused by computing the `log` of small inputs.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self)
    func logSumExp(alongAxes axes: [Int]) -> Tensor {
        // TODO(TF-433): Remove workaround for differentiating `map`.
        let axes = withoutDerivative(at: axes) { $0.map(Int32.init) }
        return logSumExp(alongAxes: Tensor<Int32>(axes))
    }

    /// Returns `log(exp(self).sum(alongAxes: axes))`. The reduced dimensions are retained with 
    /// value `1`.
    ///
    /// This function is more numerically stable than computing
    /// `log(exp(self).sum(alongAxes: axes))` directly. It avoids overflows caused by computing 
    /// the `exp` of large inputs and underflows caused by computing the `log` of small inputs.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self)
    func logSumExp(alongAxes axes: Int...) -> Tensor {
        logSumExp(alongAxes: axes)
    }
}

/// Pair of first and second moments (i.e., mean and variance).
/// - Note: This is needed because tuple types are not differentiable.
public struct Moments<Scalar: TensorFlowFloatingPoint>: Differentiable {
    public var mean: Tensor<Scalar>
    public var variance: Tensor<Scalar>

    @differentiable
    public init(mean: Tensor<Scalar>, variance: Tensor<Scalar>) {
        self.mean = mean
        self.variance = variance
    }
}

public extension Tensor where Scalar: TensorFlowFloatingPoint {
    /// Returns the mean and variance of this tensor along the specified axes. The reduced
    /// dimensions are removed.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: `axes` must have rank `1`.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self)
    func moments(squeezingAxes axes: Tensor<Int32>) -> Moments<Scalar> {
        let mean = self.mean(alongAxes: axes)
        let variance = squaredDifference(self, mean).mean(squeezingAxes: axes)
        return Moments(
            // The following is required because `Tensor.squeezingShape(at:)` does not accept
            // `Tensor<Int32>`-valued arguments.
            mean: mean.sum(squeezingAxes: axes),
            variance: variance)
    }

    /// Returns the mean and variance of this tensor along the specified axes. The reduced
    /// dimensions are removed.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self)
    func moments(squeezingAxes axes: [Int]) -> Moments<Scalar> {
        // TODO(TF-433): Remove workaround for differentiating `map`.
        let axes = {axes.map(Int32.init)}()
        return moments(squeezingAxes: Tensor<Int32>(axes))
    }

    /// Returns the mean and variance of this tensor along the specified axes. The reduced
    /// dimensions are removed.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self)
    func moments(squeezingAxes axes: Int...) -> Moments<Scalar> {
        moments(squeezingAxes: axes)
    }

    /// Returns the mean and variance of this tensor's elements.
    @inlinable
    @differentiable(wrt: self)
    func moments() -> Moments<Scalar> {
        moments(squeezingAxes: Array(0..<shape.rank))
    }

    /// Returns the mean and variance of this tensor along the specified axes. The reduced
    /// dimensions are retained with value `1`.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: `axes` must have rank `1`.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self)
    func moments(alongAxes axes: Tensor<Int32>) -> Moments<Scalar> {
        let mean = self.mean(alongAxes: axes)
        let variance = squaredDifference(self, mean).mean(alongAxes: axes)
        return Moments<Scalar>(mean: mean, variance: variance)
    }

    /// Returns the mean and variance of this tensor along the specified axes. The reduced
    /// dimensions are retained with value `1`.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self)
    func moments(alongAxes axes: [Int]) -> Moments<Scalar> {
        // TODO(TF-433): Remove workaround for differentiating `map`.
        let axes = {axes.map(Int32.init)}()
        return moments(alongAxes: Tensor<Int32>(axes))
    }

    /// Returns the mean and variance of this tensor along the specified axes. The reduced
    /// dimensions are retained with value `1`.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self)
    func moments(alongAxes axes: Int...) -> Moments<Scalar> {
        moments(alongAxes: axes)
    }
}

//===------------------------------------------------------------------------------------------===//
// Linear Algebra
//===------------------------------------------------------------------------------------------===//

/// Performs matrix multiplication with another tensor and produces the result.
@inlinable
@differentiable(vjp: _vjpMatmul(_:transposed:_:transposed:) where Scalar: TensorFlowFloatingPoint)
public func matmul<Scalar: Numeric>(
    _ lhs: Tensor<Scalar>,
    transposed transposeLhs: Bool = false,
    _ rhs: Tensor<Scalar>,
    transposed transposeRhs: Bool = false
) -> Tensor<Scalar> {
    if lhs.rank > 2 || rhs.rank > 2 {
        // TODO(TF-629): Conjugate to make compatible with the adjoint.
        return _Raw.batchMatMulV2(lhs, rhs, adjX: transposeLhs, adjY: transposeRhs)
    }
    return _Raw.matMul(lhs, rhs, transposeA: transposeLhs, transposeB: transposeRhs)
}

@inlinable
internal func _vjpMatmul<Scalar: TensorFlowFloatingPoint>(
    _ lhs: Tensor<Scalar>,
    transposed transposeLhs: Bool = false,
    _ rhs: Tensor<Scalar>,
    transposed transposeRhs: Bool = false
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
    let value = matmul(lhs, transposed: transposeLhs, rhs, transposed: transposeRhs)
    return (value, { [lhsShape = lhs.shapeTensor, rhsShape = rhs.shapeTensor] v in
        let (lhsGrad, rhsGrad): (Tensor<Scalar>, Tensor<Scalar>)
        switch (transposeLhs, transposeRhs) {
        case (false, false):
            lhsGrad = matmul(v, transposed: false, rhs, transposed: true)
            rhsGrad = matmul(lhs, transposed: true, v, transposed: false)
        case (false, true):
            lhsGrad = matmul(v, rhs)
            rhsGrad = matmul(lhs, transposed: true, v, transposed: false)
        case (true, false):
            lhsGrad = matmul(v, transposed: false, rhs, transposed: true)
            rhsGrad = matmul(lhs, v)
        case (true, true):
            lhsGrad = matmul(v, transposed: true, rhs, transposed: true)
            rhsGrad = matmul(lhs, transposed: true, v, transposed: true)
        }
        let lhsRank = lhsShape.shape[0] - 2
        let rhsRank = rhsShape.shape[0] - 2
        let (lhsAxes, rhsAxes) = _Raw.broadcastGradientArgs(
            s0: lhsShape[..<lhsRank],
            s1: rhsShape[..<rhsRank])
        return (lhsGrad.sum(squeezingAxes: lhsAxes).reshaped(toShape: lhsShape),
                rhsGrad.sum(squeezingAxes: rhsAxes).reshaped(toShape: rhsShape))
    })
}

infix operator •: MultiplicationPrecedence

public extension Tensor where Scalar: Numeric {
    // TODO: We have to define a custom VJP on • because AD can't yet differentiate generic methods.
    // After AD can differentiate generic methods, remove the custom VJP.

    /// Performs matrix multiplication between two tensors and produces the result.
    @inlinable
    @differentiable(vjp: _vjpMatmulOperator(lhs:rhs:) where Scalar: TensorFlowFloatingPoint)
    static func • (lhs: Tensor, rhs: Tensor) -> Tensor {
        matmul(lhs, rhs)
    }
}

// TODO: We have to define a custom VJP on • because AD can't yet
// differentiate generic methods. After AD can differentiate generic methods,
// remove the custom VJP.
internal extension Tensor where Scalar: TensorFlowFloatingPoint {
    @inlinable
    static func _vjpMatmulOperator(
        lhs: Tensor,
        rhs: Tensor
    ) -> (Tensor, (Tensor) -> (Tensor, Tensor)) {
        _vjpMatmul(lhs, rhs)
    }
}

public extension Tensor where Scalar: TensorFlowFloatingPoint {
    /// Returns the QR decomposition of each inner matrix in the tensor, a tensor with inner 
    /// orthogonal matrices `q` and a tensor with inner upper triangular matrices `r`, such that the 
    /// tensor is equal to `matmul(q, r)`.
    /// 
    /// - Parameters:
    ///   - fullMatrices: If `true`, compute full-sized `q` and `r`. Otherwise compute only the 
    ///     leading `min(shape[rank - 1], shape[rank - 2])` columns of `q`.
    ///  
    func qrDecomposition(fullMatrices: Bool = false) -> (q: Tensor<Scalar>, r: Tensor<Scalar>) {
        return _Raw.qr(self, fullMatrices: fullMatrices)
    }

    /// Returns the diagonal part of the tensor.
    ///
    /// For example:
    ///
    /// ```
    /// // 't' is [[1, 0, 0, 0]
    /// //         [0, 2, 0, 0]
    /// //         [0, 0, 3, 0]
    /// //         [0, 0, 0, 4]]
    /// t.diagonalPart()
    /// // [1, 2, 3, 4]
    ///
    func diagonalPart() -> Tensor<Scalar> {
        return _Raw.diagPart(self)
    }
}
