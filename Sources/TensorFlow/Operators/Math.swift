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

infix operator .>: ComparisonPrecedence
infix operator .==: ComparisonPrecedence

// TODO:
// - Consider explicit broadcasting for elementwise binary ops when
//   scalarization and rank getter are implemented.

// TODO: Remove the following extension once `./` and `./=` are defined for
// `PointwiseMultiplicative`.

infix operator ./: MultiplicationPrecedence
infix operator ./=: AssignmentPrecedence

extension PointwiseMultiplicative {
  public static func ./ (lhs: Self, rhs: Self) -> Self {
    lhs .* rhs.reciprocal
  }

  public static func ./= (lhs: inout Self, rhs: Self) {
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
    _Raw.sqrt(x)
  }

  @inlinable
  @derivative(of: sqrt)
  internal static func _vjpSqrt(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let value = Tensor.sqrt(x)
    return (value, { v in v / (2 * value) })
  }

  /// The cosine of `x`, interpreted as an angle in radians.
  @differentiable
  public static func cos(_ x: Self) -> Self {
    _Raw.cos(x)
  }

  @inlinable
  @derivative(of: cos)
  internal static func _vjpCos(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (cos(x), { v in -v * sin(x) })
  }

  /// The sine of `x`, interpreted as an angle in radians.
  @differentiable
  public static func sin(_ x: Self) -> Self {
    _Raw.sin(x)
  }

  @inlinable
  @derivative(of: sin)
  internal static func _vjpSin(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (sin(x), { v in v * cos(x) })
  }

  /// The tangent of `x`, interpreted as an angle in radians.
  @differentiable
  public static func tan(_ x: Self) -> Self {
    _Raw.tan(x)
  }

  @inlinable
  @derivative(of: tan)
  internal static func _vjpTan(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let value = tan(x)
    return (value, { v in v * (1 + value.squared()) })
  }

  /// The inverse cosine of `x` in radians.
  @differentiable
  public static func acos(_ x: Self) -> Self {
    _Raw.acos(x)
  }

  @inlinable
  @derivative(of: acos)
  internal static func _vjpAcos(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (acos(x), { v in -v / sqrt(1 - x.squared()) })
  }

  /// The inverse sine of `x` in radians.
  @differentiable
  public static func asin(_ x: Self) -> Self {
    _Raw.asin(x)
  }

  @inlinable
  @derivative(of: asin)
  internal static func _vjpAsin(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (asin(x), { v in v / sqrt(1 - x.squared()) })
  }

  /// The inverse tangent of `x` in radians.
  @differentiable
  public static func atan(_ x: Self) -> Self {
    _Raw.atan(x)
  }

  @inlinable
  @derivative(of: atan)
  internal static func _vjpAtan(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (atan(x), { v in v / (1 + x.squared()) })
  }

  /// The hyperbolic cosine of `x`.
  @differentiable
  public static func cosh(_ x: Self) -> Self {
    _Raw.cosh(x)
  }

  @inlinable
  @derivative(of: cosh)
  internal static func _vjpCosh(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (cosh(x), { v in v * sinh(x) })
  }

  /// The hyperbolic sine of `x`.
  @differentiable
  public static func sinh(_ x: Self) -> Self {
    _Raw.sinh(x)
  }

  @inlinable
  @derivative(of: sinh)
  internal static func _vjpSinh(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (sinh(x), { v in v * cosh(x) })
  }

  /// The hyperbolic tangent of `x`.
  @differentiable
  public static func tanh(_ x: Self) -> Self {
    _Raw.tanh(x)
  }

  @inlinable
  @derivative(of: tanh)
  internal static func _vjpTanh(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let value = tanh(x)
    return (value, { v in v * (1 - value.squared()) })
  }

  /// The inverse hyperbolic cosine of `x`.
  @differentiable
  public static func acosh(_ x: Self) -> Self {
    _Raw.acosh(x)
  }

  @inlinable
  @derivative(of: acosh)
  internal static func _vjpAcosh(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (acosh(x), { v in v / asinh(x) })
  }

  /// The inverse hyperbolic sine of `x`.
  @differentiable
  public static func asinh(_ x: Self) -> Self {
    _Raw.asinh(x)
  }

  @inlinable
  @derivative(of: asinh)
  internal static func _vjpAsinh(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (asinh(x), { v in v / acosh(x) })
  }

  /// The inverse hyperbolic tangent of `x`.
  @differentiable
  public static func atanh(_ x: Self) -> Self {
    _Raw.atanh(x)
  }

  @inlinable
  @derivative(of: atanh)
  internal static func _vjpAtanh(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (atanh(x), { v in v / (1 - x.squared()) })
  }

  /// The exponential function applied to `x`, or `e**x`.
  @differentiable
  public static func exp(_ x: Self) -> Self {
    _Raw.exp(x)
  }

  @inlinable
  @derivative(of: exp)
  internal static func _vjpExp(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let value = exp(x)
    return (value, { v in value * v })
  }

  /// Two raised to to power `x`.
  @differentiable
  public static func exp2(_ x: Self) -> Self {
    pow(Tensor(2, on: x.device), x)
  }

  /// Ten raised to to power `x`.
  @differentiable
  public static func exp10(_ x: Self) -> Self {
    pow(Tensor(10, on: x.device), x)
  }

  /// `exp(x) - 1` evaluated so as to preserve accuracy close to zero.
  @differentiable
  public static func expm1(_ x: Self) -> Self {
    _Raw.expm1(x)
  }

  @inlinable
  @derivative(of: expm1)
  internal static func _vjpExpm1(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let y = expm1(x)
    return (y, { v in v * y })
  }

  /// The natural logarithm of `x`.
  @differentiable
  public static func log(_ x: Self) -> Self {
    _Raw.log(x)
  }

  @inlinable
  @derivative(of: log)
  internal static func _vjpLog(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (log(x), { v in v / x })
  }

  /// The base-two logarithm of `x`.
  @differentiable
  public static func log2(_ x: Self) -> Self {
    log(x) / Scalar.log(2)
  }

  /// The base-ten logarithm of `x`.
  @differentiable
  public static func log10(_ x: Self) -> Self {
    log(x) / Scalar.log(10)
  }

  /// `log(1 + x)` evaluated so as to preserve accuracy close to zero.
  @differentiable
  public static func log1p(_ x: Self) -> Self {
    _Raw.log1p(x)
  }

  @inlinable
  @derivative(of: log1p)
  internal static func _vjpLog1p(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (log1p(x), { v in _Raw.xdivy(v, 1 + x) })
  }

  /// `exp(y log(x))` computed without loss of intermediate precision.
  ///
  /// For real types, if `x` is negative the result is NaN, even if `y` has
  /// an integral value. For complex types, there is a branch cut on the
  /// negative real axis.
  @differentiable
  public static func pow(_ x: Self, _ y: Self) -> Self {
    _Raw.pow(x, y)
  }

  @inlinable
  @derivative(of: pow)
  internal static func _vjpPow(
    _ x: Tensor, _ y: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
    let value = pow(x, y)
    return (
      value,
      { v in
        let safeX = x.replacing(with: Tensor(onesLike: x), where: x .<= 0)
        let lhsGrad = v * y * pow(x, y - 1)
        let rhsGrad = value * v * log(safeX)
        let (lhsShape, rhsShape) = (x.shapeTensor, y.shapeTensor)
        let (lhsAxes, rhsAxes) = _Raw.broadcastGradientArgs(s0: lhsShape, s1: rhsShape)
        return (
          lhsGrad.sum(squeezingAxes: lhsAxes).reshaped(toShape: lhsShape),
          rhsGrad.sum(squeezingAxes: rhsAxes).reshaped(toShape: rhsShape)
        )
      }
    )
  }

  /// `x` raised to the `n`th power.
  ///
  /// The product of `n` copies of `x`.
  @differentiable
  public static func pow(_ x: Self, _ n: Int) -> Self {
    pow(x, Tensor(Scalar(n), on: x.device))
  }

  /// The `n`th root of `x`.
  ///
  /// For real types, if `x` is negative and `n` is even, the result is NaN.
  /// For complex types, there is a branch cut along the negative real axis.
  @differentiable
  public static func root(_ x: Self, _ n: Int) -> Self {
    sign(x) * pow(abs(x), Tensor(Scalar(1) / Scalar(n), on: x.device))
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

// Note: previously, these `VectorProtocol` operator definitions were internal.
// This was confusing to library users, since operators were unavailable.
//
// Publicly exposing these operator definitions is currently problematic due to
// the negative impact on operator type-checking performance:
// https://github.com/apple/swift/pull/29815
//
// Consider publicly exposing these operators when tensorflow/swift-apis is no
// longer built as part of the Swift standard library,
/*
public extension VectorProtocol {
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

public extension VectorProtocol where VectorSpaceScalar: SignedNumeric {
    static prefix func - (x: Self) -> Self {
        .zero - x
    }

    static func - (lhs: VectorSpaceScalar, rhs: Self) -> Self {
        (-rhs).adding(lhs)
    }
}
*/

//===------------------------------------------------------------------------------------------===//
// Additional Element-wise Operators
//===------------------------------------------------------------------------------------------===//

extension Tensor where Scalar: Numeric {
  /// Adds the scalar to every scalar of the tensor and produces the sum.
  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public static func + (lhs: Scalar, rhs: Tensor) -> Tensor {
    return Tensor(lhs, deviceAndPrecisionLike: rhs) + rhs
  }

  /// Adds the scalar to every scalar of the tensor and produces the sum.
  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public static func + (lhs: Tensor, rhs: Scalar) -> Tensor {
    return lhs + Tensor(rhs, deviceAndPrecisionLike: lhs)
  }

  /// Subtracts the scalar from every scalar of the tensor and produces the difference.
  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public static func - (lhs: Scalar, rhs: Tensor) -> Tensor {
    return Tensor(lhs, deviceAndPrecisionLike: rhs) - rhs
  }

  /// Subtracts the scalar from every scalar of the tensor and produces the difference
  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public static func - (lhs: Tensor, rhs: Scalar) -> Tensor {
    return lhs - Tensor(rhs, deviceAndPrecisionLike: lhs)
  }

  /// Adds two tensors and stores the result in the left-hand-side variable.
  /// - Note: `+=` supports broadcasting.
  @inlinable
  public static func += (lhs: inout Tensor, rhs: Tensor) {
    lhs = lhs + rhs
  }

  /// Adds the scalar to every scalar of the tensor and stores the result in the left-hand-side
  /// variable.
  @inlinable
  public static func += (lhs: inout Tensor, rhs: Scalar) {
    lhs = lhs + rhs
  }

  /// Subtracts the second tensor from the first and stores the result in the left-hand-side
  /// variable.
  /// - Note: `-=` supports broadcasting.
  @inlinable
  public static func -= (lhs: inout Tensor, rhs: Tensor) {
    lhs = lhs - rhs
  }

  /// Subtracts the scalar from every scalar of the tensor and stores the result in the
  /// left-hand-side variable.
  @inlinable
  public static func -= (lhs: inout Tensor, rhs: Scalar) {
    lhs = lhs - rhs
  }

  /// Returns the tensor produced by multiplying the two tensors.
  /// - Note: `*` supports broadcasting.
  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public static func * (lhs: Tensor, rhs: Tensor) -> Tensor {
    return _Raw.mul(lhs, rhs)
  }

  /// Returns the tensor by multiplying it with every scalar of the tensor.
  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public static func * (lhs: Scalar, rhs: Tensor) -> Tensor {
    return Tensor(lhs, deviceAndPrecisionLike: rhs) * rhs
  }

  /// Multiplies the scalar with every scalar of the tensor and produces the product.
  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public static func * (lhs: Tensor, rhs: Scalar) -> Tensor {
    return lhs * Tensor(rhs, deviceAndPrecisionLike: lhs)
  }

  /// Multiplies two tensors and stores the result in the left-hand-side variable.
  /// - Note: `*=` supports broadcasting.
  @inlinable
  public static func *= (lhs: inout Tensor, rhs: Tensor) {
    lhs = lhs * rhs
  }

  /// Multiplies the tensor with the scalar, broadcasting the scalar, and stores the result in the
  /// left-hand-side variable.
  @inlinable
  public static func *= (lhs: inout Tensor, rhs: Scalar) {
    lhs = lhs * rhs
  }

  /// Returns the quotient of dividing the first tensor by the second.
  /// - Note: `/` supports broadcasting.
  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public static func / (lhs: Tensor, rhs: Tensor) -> Tensor {
    return _Raw.div(lhs, rhs)
  }

  /// Returns the quotient of dividing the scalar by the tensor, broadcasting the scalar.
  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public static func / (lhs: Scalar, rhs: Tensor) -> Tensor {
    return Tensor(lhs, deviceAndPrecisionLike: rhs) / rhs
  }

  /// Returns the quotient of dividing the tensor by the scalar, broadcasting the scalar.
  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public static func / (lhs: Tensor, rhs: Scalar) -> Tensor {
    return lhs / Tensor(rhs, deviceAndPrecisionLike: lhs)
  }

  /// Divides the first tensor by the second and stores the quotient in the left-hand-side
  /// variable.
  @inlinable
  public static func /= (lhs: inout Tensor, rhs: Tensor) {
    lhs = lhs / rhs
  }

  /// Divides the tensor by the scalar, broadcasting the scalar, and stores the quotient in the
  /// left-hand-side variable.
  @inlinable
  public static func /= (lhs: inout Tensor, rhs: Scalar) {
    lhs = lhs / rhs
  }

  /// Returns the remainder of dividing the first tensor by the second.
  /// - Note: `%` supports broadcasting.
  @inlinable
  public static func % (lhs: Tensor, rhs: Tensor) -> Tensor {
    return _Raw.mod(lhs, rhs)
  }

  /// Returns the remainder of dividing the tensor by the scalar, broadcasting the scalar.
  @inlinable
  public static func % (lhs: Tensor, rhs: Scalar) -> Tensor {
    return lhs % Tensor(rhs, deviceAndPrecisionLike: lhs)
  }

  /// Returns the remainder of dividing the scalar by the tensor, broadcasting the scalar.
  @inlinable
  public static func % (lhs: Scalar, rhs: Tensor) -> Tensor {
    return Tensor(lhs, deviceAndPrecisionLike: rhs) % rhs
  }

  /// Divides the first tensor by the second and stores the remainder in the left-hand-side
  /// variable.
  @inlinable
  public static func %= (lhs: inout Tensor, rhs: Tensor) {
    lhs = lhs % rhs
  }

  /// Divides the tensor by the scalar and stores the remainder in the left-hand-side variable.
  @inlinable
  public static func %= (lhs: inout Tensor, rhs: Scalar) {
    lhs = lhs % rhs
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: +)
  static func _vjpAdd(lhs: Tensor, rhs: Scalar) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Scalar)
  ) {
    return (lhs + rhs, { v in (v, v.sum().scalarized()) })
  }

  @inlinable
  @derivative(of: +)
  static func _vjpAdd(lhs: Scalar, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Scalar, Tensor)
  ) {
    return (lhs + rhs, { v in (v.sum().scalarized(), v) })
  }

  @inlinable
  @derivative(of: -)
  static func _vjpSubtract(lhs: Tensor, rhs: Scalar) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Scalar)
  ) {
    return (lhs - rhs, { v in (v, -v.sum().scalarized()) })
  }

  @inlinable
  @derivative(of: -)
  static func _vjpSubtract(lhs: Scalar, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Scalar, Tensor)
  ) {
    return (lhs - rhs, { v in (v.sum().scalarized(), -v) })
  }

  @inlinable
  @derivative(of: *)
  static func _vjpMultiply(lhs: Tensor, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    return (
      lhs * rhs,
      { [broadcastPb = BroadcastingPullback(lhs, rhs)] v in
        return broadcastPb(rhs * v, lhs * v)
      }
    )
  }

  @inlinable
  @derivative(of: *)
  static func _vjpMultiply(lhs: Tensor, rhs: Scalar) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Scalar)
  ) {
    return (lhs * rhs, { v in (v * rhs, (v * lhs).sum().scalarized()) })
  }

  @inlinable
  @derivative(of: *)
  static func _vjpMultiply(lhs: Scalar, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Scalar, Tensor)
  ) {
    return (lhs * rhs, { v in ((v * rhs).sum().scalarized(), v * lhs) })
  }

  @inlinable
  @derivative(of: /)
  static func _vjpDivide(lhs: Tensor, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    return (
      lhs / rhs,
      { [broadcastPb = BroadcastingPullback(lhs, rhs)] v in
        return broadcastPb(v / rhs, -lhs / rhs.squared() * v)
      }
    )
  }

  @inlinable
  @derivative(of: /)
  static func _vjpDivide(lhs: Tensor, rhs: Scalar) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Scalar)
  ) {
    return (
      lhs / rhs,
      { v in
        (
          v / rhs,
          (v * -lhs / Tensor(rhs, deviceAndPrecisionLike: lhs).squared()).sum().scalarized()
        )
      }
    )
  }

  @inlinable
  @derivative(of: /)
  static func _vjpDivide(lhs: Scalar, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Scalar, Tensor)
  ) {
    return (lhs / rhs, { v in ((v / rhs).sum().scalarized(), v * -lhs / rhs.squared()) })
  }
}

// MARK: Optimized derivatives for tensor-scalar operations

// Note: these derivatives for `(Tensor, Scalar) -> Tensor` binary operations copy ones from above,
// but are differentiable only with respect to the `Tensor` argument.
//
// This avoids unnecessary work to compute the derivative with respect to `Scalar` arguments, which
// involves calling `Tensor.scalarized()` and thereby triggering X10 tensor materialization.

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: +, wrt: lhs)
  static func _vjpAdd(lhs: Tensor, rhs: Scalar) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    return (lhs + rhs, { v in v })
  }

  @inlinable
  @derivative(of: +, wrt: rhs)
  static func _vjpAdd(lhs: Scalar, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    return (lhs + rhs, { v in v })
  }

  @inlinable
  @derivative(of: -, wrt: lhs)
  static func _vjpSubtract(lhs: Tensor, rhs: Scalar) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    return (lhs - rhs, { v in v })
  }

  @inlinable
  @derivative(of: -, wrt: rhs)
  static func _vjpSubtract(lhs: Scalar, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    return (lhs - rhs, { v in -v })
  }

  @inlinable
  @derivative(of: *, wrt: lhs)
  static func _vjpMultiply(lhs: Tensor, rhs: Scalar) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    return (lhs * rhs, { v in v * rhs })
  }

  @inlinable
  @derivative(of: *, wrt: rhs)
  static func _vjpMultiply(lhs: Scalar, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    return (lhs * rhs, { v in v * lhs })
  }

  @inlinable
  @derivative(of: /, wrt: lhs)
  static func _vjpDivide(lhs: Tensor, rhs: Scalar) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    return (lhs / rhs, { v in v / rhs })
  }

  @inlinable
  @derivative(of: /, wrt: rhs)
  static func _vjpDivide(lhs: Scalar, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    return (lhs / rhs, { v in v * -lhs / rhs.squared() })
  }
}

extension Tensor where Scalar == Bool {
  /// Returns `!self` element-wise.
  @inlinable
  public func elementsLogicalNot() -> Tensor {
    return _Raw.logicalNot(self)
  }

  /// Returns `self && other` element-wise.
  /// - Note: `&&` supports broadcasting.
  @inlinable
  public func elementsLogicalAnd(_ other: Tensor) -> Tensor {
    return _Raw.logicalAnd(self, other)
  }

  /// Returns `self && other` element-wise, broadcasting `other`.
  @inlinable
  public func elementsLogicalAnd(_ other: Scalar) -> Tensor {
    return elementsLogicalAnd(Tensor(other, on: device))
  }

  /// Returns `self || other` element-wise.
  @inlinable
  public func elementsLogicalOr(_ other: Tensor) -> Tensor {
    return _Raw.logicalOr(self, other)
  }

  /// Returns `self || other` element-wise, broadcasting `other`.
  @inlinable
  public func elementsLogicalOr(_ other: Scalar) -> Tensor {
    return elementsLogicalOr(Tensor(other, on: device))
  }
}

extension Tensor where Scalar: TensorFlowNumeric {
  /// Returns `max(min(self, max), min)`.
  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public func clipped(min: Tensor, max: Tensor) -> Tensor {
    _Raw.clipByValue(t: self, clipValueMin: min, clipValueMax: max)
  }

  /// Returns `max(min(self, max), min)`.
  @inlinable
  @differentiable(wrt: (self, min) where Scalar: TensorFlowFloatingPoint)
  public func clipped(min: Tensor, max: Scalar) -> Tensor {
    clipped(min: min, max: Tensor(max, deviceAndPrecisionLike: self))
  }

  /// Returns `max(min(self, max), min)`.
  @inlinable
  @differentiable(wrt: (self, max) where Scalar: TensorFlowFloatingPoint)
  public func clipped(min: Scalar, max: Tensor) -> Tensor {
    clipped(min: Tensor(min, deviceAndPrecisionLike: self), max: max)
  }

  /// Returns `max(min(self, max), min)`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func clipped(min: Scalar, max: Scalar) -> Tensor {
    clipped(
      min: Tensor(min, deviceAndPrecisionLike: self),
      max: Tensor(max, deviceAndPrecisionLike: self))
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: clipped)
  func _vjpClipped(min: Tensor, max: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor, Tensor)
  ) {
    (
      clipped(min: min, max: max),
      { v in
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
        return (
          selfGradient.sum(squeezingAxes: selfAxes).reshaped(toShape: selfShape),
          minGradient.sum(squeezingAxes: minAxes).reshaped(toShape: minShape),
          maxGradient.sum(squeezingAxes: maxAxes).reshaped(toShape: maxShape)
        )
      }
    )
  }
}

//===------------------------------------------------------------------------------------------===//
// Element-wise Unary Math Functions
//===------------------------------------------------------------------------------------------===//

// Export Glibc/Darwin/ucrt math functions. We should not require users to import
// Foundation/Darwin/Glibc/ucrt in order to use scalar math functions.

#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
  @_exported import Darwin.C
#elseif os(Windows)
  @_exported import ucrt
#else
  @_exported import Glibc
#endif

// FIXME: Scoped imports are not yet supported in parseable module interfaces, so
// `@_exported import` won't work. When that becomes supported, switch to `@_exported import` by
// removing `import Darwin.C/Glibc` above and uncommenting the following lines.
//
// In the meantime, consider using indirect wrappers for each function to avoidleaking random libc
// symbols to users' code completion.
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
// #elseif os(Windows)
// @_exported import func ucrt.sin
// @_exported import func ucrt.cos
// @_exported import func ucrt.tan
// @_exported import func ucrt.sinf
// @_exported import func ucrt.cosf
// @_exported import func ucrt.tanf
// @_exported import func ucrt.sinh
// @_exported import func ucrt.cosh
// @_exported import func ucrt.tanh
// @_exported import func ucrt.sinhf
// @_exported import func ucrt.coshf
// @_exported import func ucrt.tanhf
// @_exported import func ucrt.log
// @_exported import func ucrt.logf
// @_exported import func ucrt.exp
// @_exported import func ucrt.expf
// @_exported import func ucrt.pow
// @_exported import func ucrt.powf
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

extension Tensor where Scalar: SignedNumeric {
  /// Returns the negation of the specified tensor element-wise.
  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public static prefix func - (rhs: Tensor) -> Tensor {
    return _Raw.neg(rhs)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: -)
  static func _vjpNegate(_ x: Tensor) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    return (-x, { v in -v })
  }
}

/// Returns the absolute value of the specified tensor element-wise.
@inlinable
@differentiable(where T: TensorFlowFloatingPoint)
public func abs<T: SignedNumeric>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.abs(x)
}

@inlinable
@derivative(of: abs)
internal func _vjpAbs<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> Tensor<T>) {
  let sign = _Raw.sign(x)
  return (abs(x), { v in v * sign })
}

/// Returns the natural logarithm of the specified tensor element-wise.
@inlinable
@differentiable
public func log<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.log(x)
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
@differentiable
public func log1p<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.log1p(x)
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
@differentiable
public func sin<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.sin(x)
}

/// Returns the cosine of the specified tensor element-wise.
@inlinable
@differentiable
public func cos<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.cos(x)
}

/// Returns the tangent of the specified tensor element-wise.
@inlinable
@differentiable
public func tan<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.tan(x)
}

/// Returns the hyperbolic sine of the specified tensor element-wise.
@inlinable
@differentiable
public func sinh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.sinh(x)
}

/// Returns the hyperbolic cosine of the specified tensor element-wise.
@inlinable
@differentiable
public func cosh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.cosh(x)
}

/// Returns the hyperbolic tangent of the specified tensor element-wise.
@inlinable
@differentiable
public func tanh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.tanh(x)
}

/// Returns the inverse cosine of the specified tensor element-wise.
@inlinable
@differentiable
public func acos<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.acos(x)
}

/// Returns the inverse sine of the specified tensor element-wise.
@inlinable
@differentiable
public func asin<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.asin(x)
}

/// Returns the inverse tangent of the specified tensor element-wise.
@inlinable
@differentiable
public func atan<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.atan(x)
}

/// Returns the inverse hyperbolic cosine of the specified tensor element-wise.
@inlinable
@differentiable
public func acosh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.acosh(x)
}

/// Returns the inverse hyperbolic sine of the specified tensor element-wise.
@inlinable
@differentiable
public func asinh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.asinh(x)
}

/// Returns the inverse hyperbolic tangent of the specified tensor element-wise.
@inlinable
@differentiable
public func atanh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.atanh(x)
}

/// Returns the square of the tensor.
extension Tensor where Scalar: Numeric {
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func squared() -> Tensor {
    _Raw.square(self)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: squared)
  func _vjpSquared() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (squared(), { 2 * self * $0 })
  }
}

/// Returns the square root of the specified tensor element-wise.
@inlinable
@differentiable
public func sqrt<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.sqrt(x)
}

/// Returns the inverse square root of the specified tensor element-wise.
@inlinable
@differentiable
public func rsqrt<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.rsqrt(x)
}

@inlinable
@derivative(of: rsqrt)
internal func _vjpRsqrt<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> Tensor<T>) {
  let value = rsqrt(x)
  return (value, { v in _Raw.rsqrtGrad(value, dy: v) })
}

/// Returns the exponential of the specified tensor element-wise.
@inlinable
@differentiable
public func exp<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.exp(x)
}

/// Returns two raised to the power of the specified tensor element-wise.
@inlinable
@differentiable
public func exp2<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.exp2(x)
}

/// Returns ten raised to the power of the specified tensor element-wise.
@inlinable
@differentiable
public func exp10<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.exp10(x)
}

/// Returns the exponential of `x - 1` element-wise.
@inlinable
@differentiable
public func expm1<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.expm1(x)
}

/// Returns the values of the specified tensor rounded to the nearest integer, element-wise.
@inlinable
@differentiable
public func round<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.round(x)
}

@inlinable
@derivative(of: round)
internal func _vjpRound<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> Tensor<T>) {
  (round(x), { v in Tensor<T>(zerosLike: v) })
}

/// Returns the ceiling of the specified tensor element-wise.
@inlinable
@differentiable
public func ceil<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.ceil(x)
}

@inlinable
@derivative(of: ceil)
internal func _vjpCeil<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> Tensor<T>) {
  (ceil(x), { _ in Tensor(zerosLike: x) })
}

/// Returns the floor of the specified tensor element-wise.
@inlinable
@differentiable
public func floor<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.floor(x)
}

@inlinable
@derivative(of: floor)
internal func _vjpFloor<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> Tensor<T>) {
  (floor(x), { v in Tensor(0, on: v.device).broadcasted(like: x) })
}

/// Returns an indication of the sign of the specified tensor element-wise.
/// Specifically, computes `y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.
@inlinable
@differentiable(where T: TensorFlowFloatingPoint)
public func sign<T: Numeric>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.sign(x)
}

@inlinable
@derivative(of: sign)
internal func _vjpSign<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> Tensor<T>) {
  (sign(x), { v in Tensor<T>(zerosLike: x) })
}

/// Returns the sigmoid of the specified tensor element-wise.
/// Specifically, computes `1 / (1 + exp(-x))`.
@inlinable
@differentiable
public func sigmoid<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.sigmoid(x)
}

@inlinable
@derivative(of: sigmoid)
internal func _vjpSigmoid<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> Tensor<T>) {
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
@differentiable
public func softplus<T: TensorFlowFloatingPoint>(_ features: Tensor<T>) -> Tensor<T> {
  _Raw.softplus(features: features)
}

@inlinable
@derivative(of: softplus)
internal func _vjpSoftplus<T: TensorFlowFloatingPoint>(
  _ features: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> Tensor<T>) {
  (softplus(features), { v in _Raw.softplusGrad(gradients: v, features: features) })
}

/// Returns the softsign of the specified tensor element-wise.
/// Specifically, computes `features/ (abs(features) + 1)`.
@inlinable
@differentiable
public func softsign<T: TensorFlowFloatingPoint>(_ features: Tensor<T>) -> Tensor<T> {
  _Raw.softsign(features: features)
}

@inlinable
@derivative(of: softsign)
internal func _vjpSoftsign<T: TensorFlowFloatingPoint>(
  _ features: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> Tensor<T>) {
  (softsign(features), { v in _Raw.softsignGrad(gradients: v, features: features) })
}

/// Returns the softmax of the specified tensor along the last axis.
/// Specifically, computes `exp(x) / exp(x).sum(alongAxes: -1)`.
@inlinable
@differentiable
public func softmax<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.softmax(logits: x)
}

/// Returns the softmax of the specified tensor along the specified axis.
/// Specifically, computes `exp(x) / exp(x).sum(alongAxes: axis)`.
@inlinable
@differentiable
public func softmax<T: TensorFlowFloatingPoint>(_ x: Tensor<T>, alongAxis axis: Int) -> Tensor<T> {
  let xExp = exp(x)
  return xExp / xExp.sum(alongAxes: Tensor<Int32>(Int32(axis), on: xExp.device))
}

@inlinable
@derivative(of: softmax)
func _vjpSoftmax<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> Tensor<T>) {
  let value = softmax(x)
  return (
    value,
    { v in
      let sumChannels = (v * value).sum(alongAxes: -1)
      return (v - sumChannels) * value
    }
  )
}

/// Returns the log-softmax of the specified tensor element-wise.
@inlinable
@differentiable
public func logSoftmax<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.logSoftmax(logits: x)
}

@inlinable
@derivative(of: logSoftmax)
func _vjpLogSoftmax<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> Tensor<T>) {
  let value = logSoftmax(x)
  return (value, { v in v - v.sum(alongAxes: -1) * exp(value) })
}

/// Returns a tensor by applying an exponential linear unit.
/// Specifically, computes `exp(x) - 1` if < 0, `x` otherwise.
/// See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
/// ](http://arxiv.org/abs/1511.07289)
@inlinable
@differentiable
public func elu<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.elu(features: x)
}

@inlinable
@derivative(of: elu)
func _vjpElu<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> Tensor<T>) {
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
  let ratio1 = Tensor<T>(0.7978845608, deviceAndPrecisionLike: x)  // An approximation of √(2/π).
  // An approximation of the Gauss error function.
  // NOTE: This is needed because the compiler otherwise gives an "unable to type-check this
  // in reasonable time" error when the below expressions are written on a single line.
  let ratio2 = Tensor<T>(0.044715, deviceAndPrecisionLike: x)
  let half = Tensor<T>(0.5, deviceAndPrecisionLike: x)
  let one = Tensor<T>(1, deviceAndPrecisionLike: x)
  let three = Tensor<T>(3, deviceAndPrecisionLike: x)
  let approximateErf = tanh(ratio1 * (x + ratio2 * pow(x, three)))
  let cdf = half * (one + approximateErf)
  return x * cdf
}

/// Returns a tensor by applying the ReLU activation function to the specified tensor element-wise.
/// Specifically, computes `max(0, x)`.
@inlinable
@differentiable
public func relu<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.relu(features: x)
}

@inlinable
@derivative(of: relu)
func _vjpRelu<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> Tensor<T>) {
  (relu(x), { v in _Raw.reluGrad(gradients: v, features: x) })
}

/// Returns a tensor by applying the ReLU6 activation function, namely `min(max(0, x), 6)`.
@inlinable
@differentiable
public func relu6<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.relu6(features: x)
}

@inlinable
@derivative(of: relu6)
func _vjpRelu6<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> Tensor<T>) {
  (relu6(x), { v in _Raw.relu6Grad(gradients: v, features: x) })
}

/// Returns a tensor by applying the leaky ReLU activation function
/// to the specified tensor element-wise.
/// Specifically, computes `max(x, x * alpha)`.
@inlinable
@differentiable(wrt: x)
public func leakyRelu<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>,
  alpha: Double = 0.2
) -> Tensor<T> {
  _Raw.leakyRelu(features: x, alpha: alpha)
}

@inlinable
@derivative(of: leakyRelu, wrt: x)
func _vjpLeakyRelu<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>,
  alpha: Double
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> Tensor<T>) {
  (
    leakyRelu(x, alpha: alpha),
    { v in
      _Raw.leakyReluGrad(gradients: v, features: x, alpha: alpha)
    }
  )
}

/// Returns a tensor by applying the SeLU activation function, namely
/// `scale * alpha * (exp(x) - 1)` if `x < 0`, and `scale * x` otherwise.
///
/// - Note: This is designed to be used together with the variance scaling layer initializers.
///   Please refer to [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515) for more
///   information.
@inlinable
@differentiable
public func selu<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.selu(features: x)
}

@inlinable
@derivative(of: selu)
func _vjpSelu<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> Tensor<T>) {
  let result = selu(x)
  return (
    result,
    { v in
      _Raw.seluGrad(gradients: v, outputs: result)
    }
  )
}

/// Returns a tensor by applying the swish activation function, namely
/// `x * sigmoid(x)`.
///
/// Source: "Searching for Activation Functions" (Ramachandran et al. 2017)
/// https://arxiv.org/abs/1710.05941
@inlinable
@differentiable
public func swish<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  x * sigmoid(x)
}

// Note: A custom vjp function for swish is required to avoid excessive
// tensor memory consumption due to storing both `x` and `sigmoid(x)` for
// backprop. This vjp recomputes `sigmoid(x)` during backprop, so that
// the `sigmoid(x)` expression can be freed during the forward pass.
@inlinable
@derivative(of: swish)
func _vjpSwish<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> Tensor<T>) {
  return (
    swish(x),
    { v in
      let sigmoidFeatures = sigmoid(x)
      let grad = sigmoidFeatures * (1.0 + x * (1 - sigmoidFeatures))
      return grad * v
    }
  )
}

/// Returns a tensor by applying the hard sigmoid activation function, namely
/// `Relu6(x+3)/6`.
///
/// Source: "Searching for MobileNetV3" (Howard et al. 2019)
/// https://arxiv.org/abs/1905.02244
@inlinable
@differentiable
public func hardSigmoid<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  relu6(x + 3) / 6.0
}

/// Returns a tensor by applying the hard swish activation function, namely
/// `x * Relu6(x+3)/6`.
///
/// Source: "Searching for MobileNetV3" (Howard et al. 2019)
/// https://arxiv.org/abs/1905.02244
@inlinable
@differentiable
public func hardSwish<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  x * hardSigmoid(x)
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  /// Returns a boolean tensor indicating which elements of `x` are finite.
  @inlinable public var isFinite: Tensor<Bool> { _Raw.isFinite(self) }

  /// Returns a boolean tensor indicating which elements of `x` are infinite.
  @inlinable public var isInfinite: Tensor<Bool> { _Raw.isInf(self) }

  /// Returns a boolean tensor indicating which elements of `x` are NaN-valued.
  @inlinable public var isNaN: Tensor<Bool> { _Raw.isNan(self) }
}

//===------------------------------------------------------------------------------------------===//
// Element-wise Binary Math Functions
//===------------------------------------------------------------------------------------------===//

/// Returns the power of the first tensor to the second tensor.
@inlinable
@differentiable
public func pow<T: TensorFlowFloatingPoint>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
  Tensor.pow(lhs, rhs)
}

/// Returns the power of the scalar to the tensor, broadcasting the scalar.
@inlinable
@differentiable(wrt: rhs where T: TensorFlowFloatingPoint)
public func pow<T: TensorFlowFloatingPoint>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T> {
  pow(Tensor(lhs, deviceAndPrecisionLike: rhs), rhs)
}

/// Returns the power of the tensor to the scalar, broadcasting the scalar.
@inlinable
@differentiable(wrt: lhs where T: TensorFlowFloatingPoint)
public func pow<T: TensorFlowFloatingPoint>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T> {
  pow(lhs, Tensor(rhs, deviceAndPrecisionLike: lhs))
}

/// Returns the power of the tensor to the scalar, broadcasting the scalar.
@inlinable
@differentiable
public func pow<T: TensorFlowFloatingPoint>(_ x: Tensor<T>, _ n: Int) -> Tensor<T> {
  pow(x, Tensor(T(n), deviceAndPrecisionLike: x))
}

/// Returns the element-wise `n`th root of the tensor.
@inlinable
@differentiable
public func root<T: TensorFlowFloatingPoint>(_ x: Tensor<T>, _ n: Int) -> Tensor<T> {
  Tensor.root(x, n)
}

/// Returns the squared difference between `x` and `y`.
/// - Returns: `(x - y) ^ 2`.
@inlinable
@differentiable(where T: TensorFlowFloatingPoint)
public func squaredDifference<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T> {
  _Raw.squaredDifference(x, y)
}

@inlinable
@derivative(of: squaredDifference)
internal func _vjpSquaredDifference<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> (Tensor<T>, Tensor<T>)) {
  (
    squaredDifference(x, y),
    { seed in
      let lhsGrad = 2 * seed * (x - y)
      return BroadcastingPullback(x, y)(lhsGrad, -lhsGrad)
    }
  )
}

/// Returns the element-wise maximum of two tensors.
/// - Note: `max` supports broadcasting.
@inlinable
@differentiable(where T: TensorFlowFloatingPoint)
public func max<T>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable {
  _Raw.maximum(lhs, rhs)
}

@inlinable
@derivative(of: max)
internal func _vjpMax<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> (Tensor<T>, Tensor<T>)) {
  let value = max(x, y)
  return (
    value,
    { v in
      _vjpMinMaxHelper(x, y, originalValue: value, seed: v, comparisonOperation: .>=)
    }
  )
}

/// Returns the element-wise maximum of the scalar and the tensor, broadcasting the scalar.
@inlinable
@differentiable(wrt: rhs where T: TensorFlowFloatingPoint)
public func max<T>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable {
  max(Tensor(lhs, deviceAndPrecisionLike: rhs), rhs)
}

/// Returns the element-wise maximum of the scalar and the tensor, broadcasting the scalar.
@inlinable
@differentiable(wrt: lhs where T: TensorFlowFloatingPoint)
public func max<T>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T> where T: Numeric & Comparable {
  max(lhs, Tensor(rhs, deviceAndPrecisionLike: lhs))
}

/// Returns the element-wise minimum of two tensors.
/// - Note: `min` supports broadcasting.
@inlinable
@differentiable(where T: TensorFlowFloatingPoint)
public func min<T>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable {
  _Raw.minimum(lhs, rhs)
}

@inlinable
@derivative(of: min)
internal func _vjpMin<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> (Tensor<T>, Tensor<T>)) {
  let value = min(x, y)
  return (
    value,
    { v in
      _vjpMinMaxHelper(x, y, originalValue: value, seed: v, comparisonOperation: .<=)
    }
  )
}

/// Returns the element-wise minimum of the scalar and the tensor, broadcasting the scalar.
@inlinable
@differentiable(wrt: rhs where T: TensorFlowFloatingPoint)
public func min<T>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable {
  min(Tensor(lhs, deviceAndPrecisionLike: rhs), rhs)
}

/// Returns the element-wise minimum of the scalar and the tensor, broadcasting the scalar.
@inlinable
@differentiable(wrt: lhs where T: TensorFlowFloatingPoint)
public func min<T>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T> where T: Numeric & Comparable {
  min(lhs, Tensor(rhs, deviceAndPrecisionLike: lhs))
}

// Note: adapted from `_MinOrMaxGrad`:
// https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/math_grad.py#L223.
@inlinable
internal func _vjpMinMaxHelper<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>,
  _ y: Tensor<T>,
  originalValue: Tensor<T>,
  seed: Tensor<T>,
  comparisonOperation: (Tensor<T>, Tensor<T>) -> Tensor<Bool>
) -> (value: Tensor<T>, pullback: Tensor<T>) {
  let mask = Tensor<T>(comparisonOperation(x, y))
  let lhsGrad = seed * mask
  let rhsGrad = seed * (1 - mask)
  let (lhsShape, rhsShape) = (x.shapeTensor, y.shapeTensor)
  let (lhsAxes, rhsAxes) = _Raw.broadcastGradientArgs(s0: lhsShape, s1: rhsShape)
  return (
    lhsGrad.sum(squeezingAxes: lhsAxes).reshaped(toShape: lhsShape),
    rhsGrad.sum(squeezingAxes: rhsAxes).reshaped(toShape: rhsShape)
  )
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

extension Tensor {
  /// Replaces elements of this tensor with `other` in the lanes where `mask` is
  /// `true`.
  ///
  /// - Precondition: `self` and `other` must have the same shape. If
  ///   `self` and `other` are scalar, then `mask` must also be scalar. If
  ///   `self` and `other` have rank greater than or equal to `1`, then `mask`
  ///   must be either have the same shape as `self` or be a 1-D `Tensor` such
  ///   that `mask.scalarCount == self.shape[0]`.
  @inlinable
  @differentiable(wrt: (self, other) where Scalar: TensorFlowFloatingPoint)
  public func replacing(with other: Tensor, where mask: Tensor<Bool>) -> Tensor {
    precondition(self.shape == other.shape, "`self` and `other` must have the same shape.")
    return _Raw.select(condition: mask, t: other, e: self)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: replacing)
  func _vjpReplacing(
    with other: Tensor,
    where mask: Tensor<Bool>
  ) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
    return (
      replacing(with: other, where: mask),
      { v in
        let zeros = Tensor(zerosLike: v)
        return (v.replacing(with: zeros, where: mask), zeros.replacing(with: v, where: mask))
      }
    )
  }
}

//===------------------------------------------------------------------------------------------===//
// Reduction Functions
//===------------------------------------------------------------------------------------------===//

extension Tensor where Scalar == Bool {
  /// Returns `true` if all scalars are equal to `true`. Otherwise, returns `false`.
  // NOTE: This overload is necessary, otherwise `all()` would refer to the variadic method
  // `all(squeezingAxes:)` with zero indices.
  @inlinable
  public func all() -> Bool {
    let axes = Tensor<Int32>(rangeFrom: 0, to: Int32(rank), stride: 1, on: device)
    return _Raw.all(self, reductionIndices: axes).scalarized()
  }

  /// Returns `true` if any scalars are equal to `true`. Otherwise, returns `false`.
  // NOTE: This overload is necessary, otherwise `any()` would refer to the variadic method
  // `any(squeezingAxes:)` with zero indices.
  @inlinable
  public func any() -> Bool {
    let axes = Tensor<Int32>(rangeFrom: 0, to: Int32(rank), stride: 1, on: device)
    return _Raw.any(self, reductionIndices: axes).scalarized()
  }

  /// Performs a logical AND operation along the specified axes. The reduced dimensions are
  /// removed.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  public func all(squeezingAxes axes: Int...) -> Tensor {
    ensureValid(axes: axes)
    let axes = axes.map(Int32.init)
    return _Raw.all(self, reductionIndices: Tensor<Int32>(axes, on: device), keepDims: false)
  }

  /// Performs a logical AND operation along the specified axes. The reduced dimensions are
  /// removed.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  public func any(squeezingAxes axes: Int...) -> Tensor {
    ensureValid(axes: axes)
    let axes = axes.map(Int32.init)
    return _Raw.any(self, reductionIndices: Tensor<Int32>(axes, on: device), keepDims: false)
  }

  /// Performs a logical AND operation along the specified axes. The reduced dimensions are
  /// retained with value 1.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  public func all(alongAxes axes: Int...) -> Tensor {
    ensureValid(axes: axes)
    let axes = axes.map(Int32.init)
    return _Raw.all(self, reductionIndices: Tensor<Int32>(axes, on: device), keepDims: true)
  }

  /// Performs a logical OR operation along the specified axes. The reduced
  /// dimensions are retained with value 1.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  public func any(alongAxes axes: Int...) -> Tensor {
    ensureValid(axes: axes)
    let axes = axes.map(Int32.init)
    return _Raw.any(self, reductionIndices: Tensor<Int32>(axes, on: device), keepDims: true)
  }
}

extension Tensor where Scalar: Numeric & Comparable {
  // NOTE: This overload is necessary, otherwise `min()` would refer to the variadic method
  // `min(squeezingAxes:)` with zero indices.
  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public func min() -> Tensor {
    let axes = Tensor<Int32>(rangeFrom: 0, to: Int32(rank), stride: 1, on: device)
    return min(squeezingAxes: axes)
  }

  // NOTE: This overload is necessary, otherwise `max()` would refer to the variadic method
  // `max(squeezingAxes:)` with zero indices.
  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public func max() -> Tensor {
    let axes = Tensor<Int32>(rangeFrom: 0, to: Int32(rank), stride: 1, on: device)
    return max(squeezingAxes: axes)
  }

  /// Returns the maximum values along the specified axes. The reduced dimensions are removed.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func max(squeezingAxes axes: Tensor<Int32>) -> Tensor {
    ensureValid(axes: axes)
    return _Raw.max(self, reductionIndices: axes, keepDims: false)
  }

  /// Returns the maximum values along the specified axes. The reduced dimensions are removed.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func max(squeezingAxes axes: [Int]) -> Tensor {
    // TODO(TF-433): Remove workaround for differentiating `map`.
    let axes = { axes.map(Int32.init) }()
    return max(squeezingAxes: Tensor<Int32>(axes, on: device))
  }

  /// Returns the maximum values along the specified axes. The reduced dimensions are removed.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func max(squeezingAxes axes: Int...) -> Tensor {
    max(squeezingAxes: axes)
  }

  /// Returns the minimum values along the specified axes. The reduced dimensions are removed.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func min(squeezingAxes axes: Tensor<Int32>) -> Tensor {
    ensureValid(axes: axes)
    return _Raw.min(self, reductionIndices: axes, keepDims: false)
  }

  /// Returns the minimum values along the specified axes. The reduced dimensions are removed.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func min(squeezingAxes axes: [Int]) -> Tensor {
    // TODO(TF-433): Remove workaround for differentiating `map`.
    let axes = { axes.map(Int32.init) }()
    return min(squeezingAxes: Tensor<Int32>(axes, on: device))
  }

  /// Returns the minimum values along the specified axes. The reduced dimensions are removed.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func min(squeezingAxes axes: Int...) -> Tensor {
    min(squeezingAxes: axes)
  }

  /// Returns the indices of the maximum values along the specified axes. The reduced dimensions
  /// are removed.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  public func argmax(squeezingAxis axis: Int) -> Tensor<Int32> {
    ensureValid(axes: [axis])
    return _Raw.argMax(self, dimension: Int64(axis))
  }

  /// Returns the indices of the minimum values along the specified axes. The reduced dimensions
  /// are removed.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  public func argmin(squeezingAxis axis: Int) -> Tensor<Int32> {
    ensureValid(axes: [axis])
    return _Raw.argMin(self, dimension: Tensor<Int32>(Int32(axis), on: device))
  }

  /// Returns the minimum along the specified axes. The reduced dimensions are retained with
  /// value 1.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func min(alongAxes axes: Tensor<Int32>) -> Tensor {
    ensureValid(axes: axes)
    return _Raw.min(self, reductionIndices: axes, keepDims: true)
  }

  /// Returns the minimum along the specified axes. The reduced dimensions are retained with
  /// value 1.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func min(alongAxes axes: [Int]) -> Tensor {
    // TODO(TF-433): Remove workaround for differentiating `map`.
    let axes = { axes.map(Int32.init) }()
    return min(alongAxes: Tensor<Int32>(axes, on: device))
  }

  /// Returns the minimum along the specified axes. The reduced dimensions are retained with
  /// value 1.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func min(alongAxes axes: Int...) -> Tensor {
    min(alongAxes: axes)
  }

  /// Returns the minimum along the specified axes. The reduced dimensions are retained with
  /// value 1.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func max(alongAxes axes: Tensor<Int32>) -> Tensor {
    ensureValid(axes: axes)
    return _Raw.max(self, reductionIndices: axes, keepDims: true)
  }

  /// Returns the minimum along the specified axes. The reduced dimensions are retained with
  /// value 1.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func max(alongAxes axes: [Int]) -> Tensor {
    // TODO(TF-433): Remove workaround for differentiating `map`.
    let axes = { axes.map(Int32.init) }()
    return max(alongAxes: Tensor<Int32>(axes, on: device))
  }

  /// Returns the minimum along the specified axes. The reduced dimensions are retained with
  /// value 1.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func max(alongAxes axes: Int...) -> Tensor {
    max(alongAxes: axes)
  }

  /// Returns the index of the maximum value of the flattened scalars.
  @inlinable
  public func argmax() -> Tensor<Int32> {
    flattened().argmax(squeezingAxis: 0)
  }

  /// Returns the index of the minimum value of the flattened scalars.
  @inlinable
  public func argmin() -> Tensor<Int32> {
    flattened().argmin(squeezingAxis: 0)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  // Note: adapted from `_MinOrMaxGrad`:
  // https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/math_grad.py#L223.
  @inlinable
  func _vjpMinMaxHelper(
    squeezingAxes axes: Tensor<Int32>,
    originalValue: Tensor,
    seed: Tensor
  ) -> Tensor {
    let yUnsqueezed = originalValue.expandingShape(at: axes.scalars.map { Int($0) })
    let gradientUnsqueezed = seed.expandingShape(at: axes.scalars.map { Int($0) })

    // Compute the number of selected (maximum or minimum) elements in each reduction dimension.
    // If there are multiple minimum or maximum elements then the gradient will be divided
    // between them.
    let indicators = Tensor(yUnsqueezed .== self)
    let selectedCount = indicators.sum(alongAxes: axes)

    return gradientUnsqueezed.broadcasted(toShape: self.shapeTensor) * indicators / selectedCount
  }

  @inlinable
  @derivative(of: max(squeezingAxes:))
  func _vjpMax(squeezingAxes axes: Tensor<Int32>) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = max(squeezingAxes: axes)
    return (
      result,
      { v in
        self._vjpMinMaxHelper(squeezingAxes: axes, originalValue: result, seed: v)
      }
    )
  }

  @inlinable
  @derivative(of: min(squeezingAxes:))
  func _vjpMin(squeezingAxes axes: Tensor<Int32>) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = min(squeezingAxes: axes)
    return (
      result,
      { v in
        self._vjpMinMaxHelper(squeezingAxes: axes, originalValue: result, seed: v)
      }
    )
  }

  // Note: adapted from `_MinOrMaxGrad`:
  // https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/math_grad.py#L223.
  @inlinable
  func _vjpMinMaxHelper(
    alongAxes axes: Tensor<Int32>,
    originalValue: Tensor,
    seed: Tensor
  ) -> Tensor {
    // Compute the number of selected (maximum or minimum) elements in each reduction dimension.
    // If there are multiple minimum or maximum elements then the gradient will be divided
    // between them.
    let indicators = Tensor(originalValue .== self)
    let selectedCount = indicators.sum(alongAxes: axes)
    return seed.broadcasted(toShape: self.shapeTensor) * indicators / selectedCount
  }

  @inlinable
  @derivative(of: max(alongAxes:))
  func _vjpMax(alongAxes axes: Tensor<Int32>) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = max(alongAxes: axes)
    return (
      result,
      { v in
        self._vjpMinMaxHelper(alongAxes: axes, originalValue: result, seed: v)
      }
    )
  }

  @inlinable
  @derivative(of: min(alongAxes:))
  func _vjpMin(alongAxes axes: Tensor<Int32>) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = min(alongAxes: axes)
    return (
      result,
      { v in
        self._vjpMinMaxHelper(alongAxes: axes, originalValue: result, seed: v)
      }
    )
  }
}

// MARK: - Numeric Reductions

extension Tensor where Scalar: Numeric {
  // MARK: - Sum

  /// Returns the sum along the specified axes. The reduced dimensions are removed.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func sum(squeezingAxes axes: Tensor<Int32>) -> Tensor {
    ensureValid(axes: axes)
    return _Raw.sum(self, reductionIndices: axes.scalars.map { Int64($0) }, keepDims: false)
  }

  /// Returns the sum along the specified axes. The reduced dimensions are removed.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func sum(squeezingAxes axes: [Int]) -> Tensor {
    // TODO(TF-433): Remove workaround for differentiating `map`.
    let axes = { axes.map(Int64.init) }()
    return _Raw.sum(self, reductionIndices: axes, keepDims: false)
  }

  /// Returns the sum along the specified axes. The reduced dimensions are removed.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func sum(squeezingAxes axes: Int...) -> Tensor {
    sum(squeezingAxes: axes)
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func sum() -> Tensor {
    flattened().sum(squeezingAxes: 0)
  }

  /// Returns the sum along the specified axes. The reduced dimensions are retained with value 1.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func sum(alongAxes axes: Tensor<Int32>) -> Tensor {
    ensureValid(axes: axes)
    return _Raw.sum(self, reductionIndices: axes, keepDims: true)
  }

  /// Returns the sum along the specified axes. The reduced dimensions are retained with value 1.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func sum(alongAxes axes: [Int]) -> Tensor {
    // TODO(TF-433): Remove workaround for differentiating `map`.
    let axes = { axes.map(Int64.init) }()
    return _Raw.sum(self, reductionIndices: axes, keepDims: true)
  }

  /// Returns the sum along the specified axes. The reduced dimensions are retained with value 1.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func sum(alongAxes axes: Int...) -> Tensor {
    sum(alongAxes: axes)
  }

  // MARK: - Product

  /// Returns the product along the specified axes. The reduced dimensions are removed.
  ///
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func product(squeezingAxes axes: Tensor<Int32>) -> Tensor {
    ensureValid(axes: axes)
    return _Raw.prod(self, reductionIndices: axes, keepDims: false)
  }

  /// Returns the product along the specified axes. The reduced dimensions are removed.
  ///
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func product(squeezingAxes axes: [Int]) -> Tensor {
    // TODO(TF-433): Remove workaround for differentiating `map`.
    let axes = { axes.map(Int32.init) }()
    return product(squeezingAxes: Tensor<Int32>(axes, on: device))
  }

  /// Returns the product along the specified axes. The reduced dimensions are removed.
  ///
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func product(squeezingAxes axes: Int...) -> Tensor {
    product(squeezingAxes: axes)
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func product() -> Tensor {
    flattened().product(squeezingAxes: 0)
  }

  /// Returns the product along the specified axes. The reduced dimensions are retained with
  /// value 1.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  public func product(alongAxes axes: Tensor<Int32>) -> Tensor {
    ensureValid(axes: axes)
    return _Raw.prod(self, reductionIndices: axes, keepDims: true)
  }

  /// Returns the product along the specified axes. The reduced dimensions are retained with
  /// value 1.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  public func product(alongAxes axes: [Int]) -> Tensor {
    // TODO(TF-433): Remove workaround for differentiating `map`.
    let axes = { axes.map(Int32.init) }()
    return product(alongAxes: Tensor<Int32>(axes, on: device))
  }

  /// Returns the product along the specified axes. The reduced dimensions are retained with
  /// value 1.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  public func product(alongAxes axes: Int...) -> Tensor {
    product(alongAxes: axes)
  }

  // MARK: - Mean

  /// Returns the arithmetic mean along the specified axes. The reduced dimensions are removed.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func mean(squeezingAxes axes: Tensor<Int32>) -> Tensor {
    ensureValid(axes: axes)
    return _Raw.mean(self, reductionIndices: axes, keepDims: false)
  }

  /// Returns the arithmetic mean along the specified axes. The reduced dimensions are removed.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func mean(squeezingAxes axes: [Int]) -> Tensor {
    // TODO(TF-433): Remove workaround for differentiating `map`.
    let axes = { axes.map(Int64.init) }()
    return _Raw.mean(self, reductionIndices: axes, keepDims: false)
  }

  /// Returns the arithmetic mean along the specified axes. The reduced dimensions are removed.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func mean(squeezingAxes axes: Int...) -> Tensor {
    mean(squeezingAxes: axes)
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func mean() -> Tensor {
    flattened().mean(squeezingAxes: [0])
  }

  /// Returns the arithmetic mean along the specified axes. The reduced dimensions are retained
  /// with value 1.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func mean(alongAxes axes: Tensor<Int32>) -> Tensor {
    ensureValid(axes: axes)
    return _Raw.mean(self, reductionIndices: axes, keepDims: true)
  }

  /// Returns the arithmetic mean along the specified axes. The reduced dimensions are retained
  /// with value 1.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func mean(alongAxes axes: [Int]) -> Tensor {
    // TODO(TF-433): Remove workaround for differentiating `map`.
    let axes = { axes.map(Int64.init) }()
    return _Raw.mean(self, reductionIndices: axes, keepDims: true)
  }

  /// Returns the arithmetic mean along the specified axes. The reduced dimensions are retained
  /// with value 1.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func mean(alongAxes axes: Int...) -> Tensor {
    mean(alongAxes: axes)
  }

  // MARK: - Variance

  /// Returns the variance along the specified axes. The reduced dimensions are removed. Does not
  /// apply Bessel's correction.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func variance(squeezingAxes axes: Tensor<Int32>) -> Tensor {
    ensureValid(axes: axes)
    let squaredDiff = squaredDifference(self, mean(alongAxes: axes))
    return squaredDiff.mean(squeezingAxes: axes)
  }

  /// Returns the variance along the specified axes. The reduced dimensions are removed. Does not
  /// apply Bessel's correction.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func variance(squeezingAxes axes: [Int]) -> Tensor {
    // TODO(TF-433): Remove workaround for differentiating `map`.
    let axes = { axes.map(Int32.init) }()
    return variance(squeezingAxes: Tensor<Int32>(axes, on: device))
  }

  /// Returns the variance along the specified axes. The reduced dimensions are retained with
  /// value 1. Does not apply Bessel's correction.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func variance(squeezingAxes axes: Int...) -> Tensor {
    variance(squeezingAxes: axes)
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func variance() -> Tensor {
    let mean = self.mean()
    let squaredDiff = squaredDifference(self, mean)
    return squaredDiff.mean()
  }

  /// Returns the variance along the specified axes. The reduced dimensions are retained with
  /// value 1. Does not apply Bessel's correction.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func variance(alongAxes axes: Tensor<Int32>) -> Tensor {
    ensureValid(axes: axes)
    let squaredDiff = squaredDifference(self, mean(alongAxes: axes))
    return squaredDiff.mean(alongAxes: axes)
  }

  /// Returns the variance along the specified axes. The reduced dimensions are retained with
  /// value 1. Does not apply Bessel's correction.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func variance(alongAxes axes: [Int]) -> Tensor {
    // TODO(TF-433): Remove workaround for differentiating `map`.
    let axes = { axes.map(Int32.init) }()
    return variance(alongAxes: Tensor<Int32>(axes, on: device))
  }

  /// Returns the variance along the specified axes. The reduced dimensions are retained with
  /// value 1. Does not apply Bessel's correction.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func variance(alongAxes axes: Int...) -> Tensor {
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
  public func cumulativeSum(
    alongAxis axis: Int,
    exclusive: Bool = false,
    reverse: Bool = false
  ) -> Tensor {
    cumulativeSum(
      alongAxis: Tensor<Int32>(Int32(axis), on: device),
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
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func cumulativeSum(
    alongAxis axis: Tensor<Int32>,
    exclusive: Bool = false,
    reverse: Bool = false
  ) -> Tensor {
    ensureValid(axes: axis)
    return _Raw.cumsum(self, axis: axis, exclusive: exclusive, reverse: reverse)
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
  public func cumulativeProduct(
    alongAxis axis: Int,
    exclusive: Bool = false,
    reverse: Bool = false
  ) -> Tensor {
    cumulativeProduct(
      alongAxis: Tensor<Int32>(Int32(axis), on: device),
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
  /// - Precondition: `axis` must have rank `0`.
  /// - Precondition: `axis` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func cumulativeProduct(
    alongAxis axis: Tensor<Int32>,
    exclusive: Bool = false,
    reverse: Bool = false
  ) -> Tensor {
    ensureValid(axes: axis)
    return _Raw.cumprod(self, axis: axis, exclusive: exclusive, reverse: reverse)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: sum(alongAxes:))
  func _vjpSum(alongAxes axes: Tensor<Int32>) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    return _vjpSum(alongAxes: axes.scalars.map { Int($0) })
  }

  @inlinable
  @derivative(of: sum(squeezingAxes:))
  func _vjpSum(squeezingAxes axes: Tensor<Int32>) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    return _vjpSum(squeezingAxes: axes.scalars.map { Int($0) })
  }

  @inlinable
  @derivative(of: sum(alongAxes:))
  func _vjpSum(alongAxes axes: [Int]) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let value = sum(alongAxes: axes)
    let m = shape.dimensions.map { Int64($0) }
    return (value, { _Raw.broadcastTo($0, shape: m) })
  }

  @inlinable
  @derivative(of: sum(squeezingAxes:))
  func _vjpSum(squeezingAxes axes: [Int]) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let value = sum(squeezingAxes: axes)
    let rank = self.rank
    return (
      value,
      { [shape = shape.dimensions.map { Int64($0) }] v in
        var expandedShape = shape
        for dim in axes { expandedShape[(dim + rank) % rank] = 1 }
        let unsqueezed = _Raw.reshape(v, shape: expandedShape)
        return _Raw.broadcastTo(unsqueezed, shape: shape)
      }
    )
  }

  @inlinable
  @derivative(of: mean(alongAxes:))
  func _vjpMean(alongAxes axes: Tensor<Int32>) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    return _vjpMean(alongAxes: axes.scalars.map { Int($0) })
  }

  @inlinable
  @derivative(of: mean(squeezingAxes:))
  func _vjpMean(squeezingAxes axes: Tensor<Int32>) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    return _vjpMean(squeezingAxes: axes.scalars.map { Int($0) })
  }

  // Specialization to avoid _Raw.gather on shapes when axes is known to be
  // [Int].
  @inlinable
  @derivative(of: mean(alongAxes:))
  func _vjpMean(alongAxes axes: [Int]) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let value = mean(alongAxes: axes)
    // Cache shape because it is a computed property.
    let cachedShape = shape
    let rank = self.rank
    let count = axes.map { cachedShape[($0 + rank) % rank] }.reduce(1, *)
    return (
      value,
      { v in
        _Raw.broadcastTo(v, shape: cachedShape.dimensions.map { Int64($0) })
          / Tensor(Scalar(count), deviceAndPrecisionLike: v)
      }
    )
  }

  // Specialization to avoid _Raw.gather on shapes when axes is known to be
  // [Int].
  @inlinable
  @derivative(of: mean(squeezingAxes:))
  func _vjpMean(squeezingAxes axes: [Int]) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let value = mean(squeezingAxes: axes)
    // Cache shape because it is a computed property.
    let cachedShape = shape
    let rank = self.rank
    let count = axes.map { cachedShape[($0 + rank) % rank] }.reduce(1, *)
    return (
      value,
      { v in
        var expandedShape = cachedShape
        for dim in axes {
          expandedShape[(dim + rank) % rank] = 1
        }
        let unsqueezed = _Raw.reshape(v, shape: expandedShape.dimensions.map { Int64($0) })
        return _Raw.broadcastTo(unsqueezed, shape: cachedShape.dimensions.map { Int64($0) })
          / Tensor(Scalar(count), deviceAndPrecisionLike: v)
      }
    )
  }

  @inlinable
  @derivative(of: cumulativeSum)
  func _vjpCumulativeSum(
    alongAxis axis: Tensor<Int32>,
    exclusive: Bool = false,
    reverse: Bool = false
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (
      cumulativeSum(alongAxis: axis, exclusive: exclusive, reverse: reverse),
      { v in
        v.cumulativeSum(alongAxis: axis, exclusive: exclusive, reverse: !reverse)
      }
    )
  }

  @inlinable
  @derivative(of: cumulativeProduct)
  func _vjpCumulativeProduct(
    alongAxis axis: Tensor<Int32>,
    exclusive: Bool = false,
    reverse: Bool = false
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = cumulativeProduct(alongAxis: axis, exclusive: exclusive, reverse: reverse)
    return (
      result,
      { v in
        (result * v).cumulativeSum(
          alongAxis: axis,
          exclusive: exclusive,
          reverse: !reverse
        ) / self
      }
    )
  }

  // Adapted from `_ProdGrad` in Python TensorFlow:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/math_grad.py
  @inlinable
  @derivative(of: product(squeezingAxes:))
  func _vjpProduct(squeezingAxes axes: Tensor<Int32>) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    // The gradient can be expressed by dividing the product by each entry of the
    // input tensor, but this approach can't deal with zeros in the input.
    // Here, we avoid this problem by composing the output as a product of two
    // `cumulativeProduct` operations.
    let result = product(squeezingAxes: axes)
    return (
      result,
      { v in
        // Reshape reduction indices for the case where the parameter is a scalar.
        var reductionIndices = axes.flattened()
        // Normalize any negative reduction indices to positive values.
        reductionIndices = (reductionIndices + Int32(self.rank)) % Int32(self.rank)

        // Expand `v` to full input shape.
        var outputShape = self.shape
        for axis in reductionIndices.scalars {
          outputShape[Int(axis)] = 1
        }
        let vBroadcasted = v.reshaped(to: outputShape).broadcasted(to: self.shape)

        // Pack all reduced dimensions into a single one, so we can perform the
        // `cumulativeProduct` operations.
        let idx = Tensor<Int32>(0..<Int32(self.rank), on: device)
        let other = Tensor<Int32>(
          Array(Set(idx.scalars).symmetricDifference(reductionIndices.scalars)), on: device)
        let perm = reductionIndices.concatenated(with: other)
        let reducedNum = Int(
          self.shapeTensor.gathering(atIndices: reductionIndices).product().scalarized())
        let otherNum = Int(
          self.shapeTensor.gathering(atIndices: other).product().scalarized())

        let permuted = self.transposed(permutation: perm)
        let reshaped = permuted.reshaped(to: [reducedNum, otherNum])
        // Calculate product, leaving out the current entry.
        let left = reshaped.cumulativeProduct(alongAxis: 0, exclusive: true, reverse: false)
        let right = reshaped.cumulativeProduct(alongAxis: 0, exclusive: true, reverse: true)
        let y = (left * right).reshaped(to: permuted.shape)

        // Invert the transpose and reshape operations.
        // Make sure to set the statically known shape information through a reshape.
        return (vBroadcasted * y.transposed(permutation: _Raw.invertPermutation(perm)))
          .reshaped(to: self.shape)
      }
    )
  }
}

// TODO: Consider making the return type be generic over `FloatingPoint` types
// so that `self`'s scalar type can be any `Numeric` type.
extension Tensor where Scalar: TensorFlowFloatingPoint {
  /// Returns the standard deviation of the elements along the specified axes. The reduced
  /// dimensions are retained with value `1`. Does not apply Bessel's correction.
  ///
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self)
  public func standardDeviation(squeezingAxes axes: Tensor<Int32>) -> Tensor {
    ensureValid(axes: axes)
    return Tensor.sqrt(variance(squeezingAxes: axes))
  }

  /// Returns the standard deviation of the elements along the specified axes. The reduced
  /// dimensions are retained with value `1`. Does not apply Bessel's correction.
  ///
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self)
  public func standardDeviation(squeezingAxes axes: [Int]) -> Tensor {
    ensureValid(axes: axes)
    return Tensor.sqrt(variance(squeezingAxes: axes))
  }

  /// Returns the standard deviation of the elements along the specified axes. The reduced
  /// dimensions are retained with value `1`. Does not apply Bessel's correction.
  ///
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self)
  public func standardDeviation(squeezingAxes axes: Int...) -> Tensor {
    standardDeviation(squeezingAxes: axes)
  }

  /// Returns the standard deviation of all elements in this tensor.
  /// Does not apply Bessel's correction.
  ///
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self)
  public func standardDeviation() -> Tensor {
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
  public func standardDeviation(alongAxes axes: Tensor<Int32>) -> Tensor {
    ensureValid(axes: axes)
    return Tensor.sqrt(variance(alongAxes: axes))
  }

  /// Returns the standard deviation of the elements along the specified axes. The reduced
  /// dimensions are retained with value `1`. Does not apply Bessel's correction.
  ///
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self)
  public func standardDeviation(alongAxes axes: [Int]) -> Tensor {
    // TODO(TF-433): Remove workaround for differentiating `map`.
    let axes = { axes.map(Int32.init) }()
    return standardDeviation(alongAxes: Tensor<Int32>(axes, on: device))
  }

  /// Returns the standard deviation of the elements along the specified axes. The reduced
  /// dimensions are retained with value `1`. Does not apply Bessel's correction.
  ///
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self)
  public func standardDeviation(alongAxes axes: Int...) -> Tensor {
    ensureValid(axes: axes)
    return Tensor.sqrt(variance(alongAxes: axes))
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
  public func logSumExp(squeezingAxes axes: Tensor<Int32>) -> Tensor {
    ensureValid(axes: axes)
    let rawMax = max(alongAxes: axes)
    let offset = withoutDerivative(at: rawMax) { rawMax in
      Tensor<Scalar>(zerosLike: rawMax).replacing(
        with: rawMax,
        where: rawMax.isFinite)
    }
    let result = Tensor.log(Tensor.exp(self - offset).sum(squeezingAxes: axes))
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
  public func logSumExp(squeezingAxes axes: [Int]) -> Tensor {
    // TODO(TF-433): Remove workaround for differentiating `map`.
    let axes = withoutDerivative(at: axes) { $0.map(Int32.init) }
    return logSumExp(squeezingAxes: Tensor<Int32>(axes, on: device))
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
  public func logSumExp(squeezingAxes axes: Int...) -> Tensor {
    logSumExp(squeezingAxes: axes)
  }

  /// Returns `log(exp(self).sum())`. The result is a scalar.
  ///
  /// This function is more numerically stable than computing `log(exp(self).sum())` directly. It
  /// avoids overflows caused by computing the `exp` of large inputs and underflows caused by
  /// computing the `log` of small inputs.
  @inlinable
  @differentiable(wrt: self)
  public func logSumExp() -> Tensor {
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
  public func logSumExp(alongAxes axes: Tensor<Int32>) -> Tensor {
    ensureValid(axes: axes)
    let rawMax = max(alongAxes: axes)
    let offset = withoutDerivative(at: rawMax) { rawMax in
      Tensor<Scalar>(zerosLike: rawMax).replacing(
        with: rawMax,
        where: rawMax.isFinite)
    }
    let result = Tensor.log(Tensor.exp(self - offset).sum(alongAxes: axes))
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
  public func logSumExp(alongAxes axes: [Int]) -> Tensor {
    // TODO(TF-433): Remove workaround for differentiating `map`.
    let axes = withoutDerivative(at: axes) { $0.map(Int32.init) }
    return logSumExp(alongAxes: Tensor<Int32>(axes, on: device))
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
  public func logSumExp(alongAxes axes: Int...) -> Tensor {
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

extension Tensor where Scalar: TensorFlowFloatingPoint {
  /// Returns the mean and variance of this tensor along the specified axes. The reduced
  /// dimensions are removed.
  ///
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: `axes` must have rank `1`.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self)
  public func moments(squeezingAxes axes: Tensor<Int32>) -> Moments<Scalar> {
    ensureValid(axes: axes)
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
  public func moments(squeezingAxes axes: [Int]) -> Moments<Scalar> {
    ensureValid(axes: axes)
    let mean = self.mean(squeezingAxes: axes)
    let variance = squaredDifference(self, mean).mean(squeezingAxes: axes)
    return Moments(mean: mean, variance: variance)
  }

  /// Returns the mean and variance of this tensor along the specified axes. The reduced
  /// dimensions are removed.
  ///
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self)
  public func moments(squeezingAxes axes: Int...) -> Moments<Scalar> {
    moments(squeezingAxes: axes)
  }

  /// Returns the mean and variance of this tensor's elements.
  @inlinable
  @differentiable(wrt: self)
  public func moments() -> Moments<Scalar> {
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
  public func moments(alongAxes axes: Tensor<Int32>) -> Moments<Scalar> {
    ensureValid(axes: axes)
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
  public func moments(alongAxes axes: [Int]) -> Moments<Scalar> {
    ensureValid(axes: axes)
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
  public func moments(alongAxes axes: Int...) -> Moments<Scalar> {
    moments(alongAxes: axes)
  }
}

//===------------------------------------------------------------------------------------------===//
// Linear Algebra
//===------------------------------------------------------------------------------------------===//

/// Performs matrix multiplication with another tensor and produces the result.
@inlinable
@differentiable(where Scalar: TensorFlowFloatingPoint)
public func matmul<Scalar: Numeric>(
  _ lhs: Tensor<Scalar>,
  transposed transposeLhs: Bool = false,
  _ rhs: Tensor<Scalar>,
  transposed transposeRhs: Bool = false
) -> Tensor<Scalar> {
  precondition(lhs.rank >= 2, "Input tensors must have at least rank 2")
  precondition(rhs.rank >= 2, "Input tensors must have at least rank 2")
  if lhs.rank > 2 || rhs.rank > 2 {
    // TODO(TF-629): Conjugate to make compatible with the adjoint.
    return _Raw.batchMatMulV2(lhs, rhs, adjX: transposeLhs, adjY: transposeRhs)
  }
  return _Raw.matMul(lhs, rhs, transposeA: transposeLhs, transposeB: transposeRhs)
}

@inlinable
@derivative(of: matmul)
internal func _vjpMatmul<Scalar: TensorFlowFloatingPoint>(
  _ lhs: Tensor<Scalar>,
  transposed transposeLhs: Bool = false,
  _ rhs: Tensor<Scalar>,
  transposed transposeRhs: Bool = false
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
  let value = matmul(lhs, transposed: transposeLhs, rhs, transposed: transposeRhs)
  return (
    value,
    { [lhsShape = lhs.shape, rhsShape = rhs.shape] v in
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
      let lhsRank = lhsShape.rank - 2
      let rhsRank = rhsShape.rank - 2
      if lhsRank == rhsRank { return (lhsGrad, rhsGrad) }
      let lhsShape = lhsShape.dimensions.map { Int64($0) }
      let rhsShape = rhsShape.dimensions.map { Int64($0) }
      let (lhsAxes, rhsAxes) = BroadcastingPullback.computeReductionAxes(
        lhsShape[..<lhsRank].map { $0 },
        rhsShape[..<rhsRank].map { $0 }
      )
      return (
        _Raw.reshape(
          _Raw.sum(lhsGrad, reductionIndices: lhsAxes, keepDims: false), shape: lhsShape),
        _Raw.reshape(_Raw.sum(rhsGrad, reductionIndices: rhsAxes, keepDims: false), shape: rhsShape)
      )
    }
  )
}

infix operator •: MultiplicationPrecedence

extension Tensor where Scalar: Numeric {
  /// Performs matrix multiplication between two tensors and produces the result.
  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public static func • (lhs: Tensor, rhs: Tensor) -> Tensor {
    matmul(lhs, rhs)
  }
}
