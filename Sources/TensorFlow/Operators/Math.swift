// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

// `pow` is defined in Darwin/Glibc on `Float` and `Double`, but there doesn't exist a generic
// version for `FloatingPoint`.
// This is a manual definition.
@inlinable
func pow<T: BinaryFloatingPoint>(_ x: T, _ y: T) -> T {
    return T(pow(Double(x), Double(y)))
}

// TODO:
// - Consider explicit broadcasting for elementwise binary ops when
//   scalarization and rank getter are implemented.

//===------------------------------------------------------------------------------------------===//
// Vector Space
//===------------------------------------------------------------------------------------------===//

extension Tensor: VectorProtocol where Scalar: Numeric {
    /// Multiplies the scalar with every scalar of the tensor and produces the product.
    @inlinable
    @differentiable(vjp: _vjpMultiply(lhs:rhs:) where Scalar: TensorFlowFloatingPoint)
    public static func * (lhs: Scalar, rhs: Tensor) -> Tensor {
        return Tensor(lhs) * rhs
    }
}

internal extension Tensor where Scalar: TensorFlowFloatingPoint {
    @inlinable
    static func _vjpMultiply(lhs: Tensor, rhs: Tensor) -> (Tensor, (Tensor) -> (Tensor, Tensor)) {
        return (lhs * rhs, { [lhsShape = lhs.shapeTensor, rhsShape = rhs.shapeTensor] v in
            let lhsGrad = rhs * v
            let rhsGrad = lhs * v
            let (lhsAxes, rhsAxes) = Raw.broadcastGradientArgs(s0: lhsShape, s1: rhsShape)
            return (lhsGrad.sum(squeezingAxes: lhsAxes).reshaped(toShape: lhsShape),
                    rhsGrad.sum(squeezingAxes: rhsAxes).reshaped(toShape: rhsShape))
        })
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

    /// Multiplies two tensors and produces their product.
    /// - Note: `*` supports broadcasting.
    @inlinable
    @differentiable(vjp: _vjpMultiply(lhs:rhs:) where Scalar: TensorFlowFloatingPoint)
    static func * (lhs: Tensor, rhs: Tensor) -> Tensor {
        return Raw.mul(lhs, rhs)
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
        return Raw.div(lhs, rhs)
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
        return Raw.mod(lhs, rhs)
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
            let (lhsAxes, rhsAxes) = Raw.broadcastGradientArgs(s0: lhsShape, s1: rhsShape)
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
    /// Computes `!self` element-wise.
    @inlinable
    func elementsLogicalNot() -> Tensor {
        return Raw.logicalNot(self)
    }

    /// Computes `self && other` element-wise.
    /// - Note: `&&` supports broadcasting.
    @inlinable
    func elementsLogicalAnd(_ other: Tensor) -> Tensor {
        return Raw.logicalAnd(self, other)
    }

    /// Computes `self && other` element-wise, broadcasting `other`.
    @inlinable
    func elementsLogicalAnd(_ other: Scalar) -> Tensor {
        return elementsLogicalAnd(Tensor(other))
    }

    /// Computes `self || other` element-wise.
    @inlinable
    func elementsLogicalOr(_ other: Tensor) -> Tensor {
        return Raw.logicalOr(self, other)
    }

    /// Computes `self || other` element-wise, broadcasting `other`.
    @inlinable
    func elementsLogicalOr(_ other: Scalar) -> Tensor {
        return elementsLogicalOr(Tensor(other))
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
    /// Computes the negation of the specified tensor element-wise.
    @inlinable
    @differentiable(vjp: _vjpNegate(_:) where Scalar: TensorFlowFloatingPoint)
    static prefix func - (rhs: Tensor) -> Tensor {
        return Raw.neg(rhs)
    }
}

internal extension Tensor where Scalar: TensorFlowFloatingPoint {
    @inlinable
    static func _vjpNegate(_ x: Tensor) -> (Tensor, (Tensor) -> Tensor) {
        return (-x, { v in -v })
    }
}

/// Computes the absolute value of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpAbs(_:) where T: TensorFlowFloatingPoint)
public func abs<T: SignedNumeric>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.abs(x)
}

@inlinable
internal func _vjpAbs<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let sign = Raw.sign(x)
    return (abs(x), { v in v * sign })
}

/// Computes the natural logarithm of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
public func log<T: FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.log(x)
}

@inlinable
internal func _vjpLog<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    return (log(x), { v in v / x })
}

/// Computes the logarithm of `1 + x` element-wise.
@inlinable
@differentiable(vjp: _vjpLog1p)
public func log1p<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    Raw.log1p(x)
}

@inlinable
func _vjpLog1p<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    (log1p(x), { v in Raw.xdivy(v, 1 + x) })
}

/// Computes `sin` of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpSin(_:) where T: TensorFlowFloatingPoint)
public func sin<T: FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.sin(x)
}

@inlinable
internal func _vjpSin<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    return (sin(x), { v in v * cos(x) })
}

/// Computes `cos` of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpCos(_:) where T: TensorFlowFloatingPoint)
public func cos<T: FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.cos(x)
}

@inlinable
internal func _vjpCos<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    return (cos(x), { v in -v * sin(x) })
}

/// Computes `tan` of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpTan(_:) where T: TensorFlowFloatingPoint)
public func tan<T: FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.tan(x)
}

@inlinable
internal func _vjpTan<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let value = tan(x)
    return (value, { v in v * (1 + value.squared()) })
}

/// Computes `sinh` of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpSinh(_:) where T: TensorFlowFloatingPoint)
public func sinh<T: FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.sinh(x)
}

@inlinable
internal func _vjpSinh<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    return (sinh(x), { v in v * cosh(x) })
}

/// Computes `cosh` of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpCosh(_:) where T: TensorFlowFloatingPoint)
public func cosh<T: FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.cosh(x)
}

@inlinable
internal func _vjpCosh<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    return (cosh(x), { v in v * sinh(x) })
}

/// Computes `tanh` of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpTanh(_:) where T: TensorFlowFloatingPoint)
public func tanh<T: FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.tanh(x)
}

@inlinable
internal func _vjpTanh<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let value = tanh(x)
    return (value, { v in v * (1 - value.squared()) })
}

/// Computes the square of the tensor.
public extension Tensor where Scalar: Numeric {
    @inlinable
    @differentiable(wrt: self, vjp: _vjpSquared() where Scalar: TensorFlowFloatingPoint)
    func squared() -> Tensor {
        return Raw.square(self)
    }
}

internal extension Tensor where Scalar: TensorFlowFloatingPoint {
    @inlinable
    func _vjpSquared() -> (Tensor, (Tensor) -> Tensor) {
        return (squared(), { 2 * self * $0 })
    }
}

/// Computes the square root of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpSqrt(_:) where T: TensorFlowFloatingPoint)
public func sqrt<T: FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.sqrt(x)
}

@inlinable
internal func _vjpSqrt<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let value = sqrt(x)
    return (value, { v in v / (2 * value) })
}

/// Computes the inverse square root of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpRsqrt(_:) where T: TensorFlowFloatingPoint)
public func rsqrt<T: FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.rsqrt(x)
}

@inlinable
internal func _vjpRsqrt<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let value = rsqrt(x)
    return (value, { v in -v / (2 * pow(x, 3 / 2)) })
}

/// Computes `exp` of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpExp(_:) where T: TensorFlowFloatingPoint)
public func exp<T: FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.exp(x)
}

@inlinable
internal func _vjpExp<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let value = exp(x)
    return (value, { v in value * v })
}

/// Computes the exponential of `x - 1` element-wise.
@inlinable
@differentiable(vjp: _vjpExpm1)
public func expm1<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    Raw.expm1(x)
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
  return Raw.round(x)
}

@inlinable
internal func _vjpRound<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
  return (round(x), { v in Tensor<T>(zerosLike: v) })
}

/// Computes the ceiling of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpCeil(_:) where T: TensorFlowFloatingPoint)
public func ceil<T: FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.ceil(x)
}

@inlinable
internal func _vjpCeil<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    return (ceil(x), { _ in Tensor(0).broadcasted(like: x) })
}

/// Computes the floor of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpFloor(_:) where T: TensorFlowFloatingPoint)
public func floor<T: FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.floor(x)
}

@inlinable
internal func _vjpFloor<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    return (floor(x), { _ in Tensor(0).broadcasted(like: x) })
}

@inlinable
@differentiable(vjp: _vjpSign(_:) where T: TensorFlowFloatingPoint)
public func sign<T: Numeric>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.sign(x)
}

@inlinable
internal func _vjpSign<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    return (sign(x), { v in Tensor<T>(zerosLike: x) })
}

/// Computes the sigmoid of the specified tensor element-wise.
/// Specifically, computes `1 / (1 + exp(-x))`.
@inlinable
@differentiable(vjp: _vjpSigmoid)
public func sigmoid<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.sigmoid(x)
}

@inlinable
internal func _vjpSigmoid<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    return (sigmoid(x), { v in Raw.sigmoidGrad(x, dy: v) })
}

/// Computes the softmax of the specified tensor along the last axis.
/// Specifically, computes `exp(x) / exp(x).sum(alongAxes: -1)`.
@inlinable
@differentiable(vjp: _vjpSoftmax(_:) where T: TensorFlowFloatingPoint)
public func softmax<T: FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.softmax(logits: x)
}

/// Computes the softmax of the specified tensor along the specified axis.
/// Specifically, computes `exp(x) / exp(x).sum(alongAxes: axis)`.
@inlinable
// TODO: [AD].
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

/// Computes the log-softmax of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpLogSoftmax(_:) where T: TensorFlowFloatingPoint)
public func logSoftmax<T: FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.logSoftmax(logits: x)
}

@inlinable
func _vjpLogSoftmax<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let value = logSoftmax(x)
    return (value, { v in v - v.sum(alongAxes: -1) * exp(value) })
}

/// Computes `relu` of the specified tensor element-wise.
/// Specifically, computes `max(0, x)`.
@inlinable
@differentiable(vjp: _vjpRelu(_:) where T: TensorFlowFloatingPoint)
public func relu<T: FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return max(0, x)
}

@inlinable
func _vjpRelu<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    return (relu(x), { v in Tensor(x .> 0) * v })
}

//===------------------------------------------------------------------------------------------===//
// Element-wise Binary Math Functions
//===------------------------------------------------------------------------------------------===//

/// Computes the power of the first tensor to the second tensor.
@inlinable
@differentiable(vjp: _vjpPow(_:_:) where T: TensorFlowFloatingPoint)
public func pow<T>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> where T: FloatingPoint {
    return Raw.pow(lhs, rhs)
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
        let (lhsAxes, rhsAxes) = Raw.broadcastGradientArgs(s0: lhsShape, s1: rhsShape)
        return (lhsGrad.sum(squeezingAxes: lhsAxes).reshaped(toShape: lhsShape),
                rhsGrad.sum(squeezingAxes: rhsAxes).reshaped(toShape: rhsShape))
    })
}

/// Computes the power of the scalar to the tensor, broadcasting the scalar.
@inlinable
// @differentiable(where T: TensorFlowFloatingPoint)
public func pow<T>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T> where T: FloatingPoint {
    return pow(Tensor(lhs), rhs)
}

/// Computes the power of the tensor to the scalar, broadcasting the scalar.
@inlinable
// @differentiable(where T: TensorFlowFloatingPoint)
public func pow<T>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T> where T: FloatingPoint {
    return pow(lhs, Tensor(rhs))
}

/// Computes the element-wise maximum of two tensors.
/// - Note: `max` supports broadcasting.
@inlinable
@differentiable(vjp: _vjpMax(_:_:) where T: TensorFlowFloatingPoint)
public func max<T>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable {
    return Raw.maximum(lhs, rhs)
}

@inlinable
internal func _vjpMax<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>, _ y: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> (Tensor<T>, Tensor<T>)) {
    let value = max(x, y)
    return (value, { v in _vjpMinMaxHelper(x, y, originalValue: value, seed: v) })
}

/// Computes the element-wise maximum of the scalar and the tensor, broadcasting the scalar.
@inlinable
// @differentiable(where T: TensorFlowFloatingPoint)
public func max<T>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable {
    return max(Tensor(lhs), rhs)
}

/// Computes the element-wise maximum of the scalar and the tensor, broadcasting the scalar.
@inlinable
// @differentiable(where T: TensorFlowFloatingPoint)
public func max<T>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T> where T: Numeric & Comparable {
    return max(lhs, Tensor(rhs))
}

/// Computes the element-wise minimum of two tensors.
/// - Note: `min` supports broadcasting.
@inlinable
@differentiable(vjp: _vjpMin(_:_:) where T: TensorFlowFloatingPoint)
public func min<T>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable {
    return Raw.minimum(lhs, rhs)
}

@inlinable
internal func _vjpMin<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>, _ y: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> (Tensor<T>, Tensor<T>)) {
    let value = min(x, y)
    return (value, { v in _vjpMinMaxHelper(x, y, originalValue: value, seed: v) })
}

/// Computes the element-wise minimum of the scalar and the tensor, broadcasting the scalar.
@inlinable
// @differentiable(where T: TensorFlowFloatingPoint)
public func min<T>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable {
    return min(Tensor(lhs), rhs)
}

/// Computes the element-wise minimum of the scalar and the tensor, broadcasting the scalar.
@inlinable
// @differentiable(where T: TensorFlowFloatingPoint)
public func min<T>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T> where T: Numeric & Comparable {
    return min(lhs, Tensor(rhs))
}

@inlinable
internal func _vjpMinMaxHelper<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>,
    _ y: Tensor<T>,
    originalValue: Tensor<T>,
    seed: Tensor<T>
) -> (Tensor<T>, Tensor<T>) {
    let denominator = 1 + Tensor<T>(x .== y)
    let lhsGrad = seed * Tensor<T>(x .== originalValue) / denominator
    let rhsGrad = seed * Tensor<T>(y .== originalValue) / denominator
    let (lhsShape, rhsShape) = (x.shapeTensor, y.shapeTensor)
    let (lhsAxes, rhsAxes) = Raw.broadcastGradientArgs(s0: lhsShape, s1: rhsShape)
    return (lhsGrad.sum(squeezingAxes: lhsAxes).reshaped(toShape: lhsShape),
            rhsGrad.sum(squeezingAxes: rhsAxes).reshaped(toShape: rhsShape))
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
        return Raw.select(condition: mask, t: other, e: self)
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
        return Raw.all(self, reductionIndices: axes).scalarized()
    }

    /// Returns `true` if any scalars are equal to `true`. Otherwise, returns `false`.
    // NOTE: This overload is necessary, otherwise `any()` would refer to the variadic method
    // `any(squeezingAxes:)` with zero indices.
    @inlinable
    func any() -> Bool {
        let axes = Tensor<Int32>(rangeFrom: 0, to: Int32(rank), stride: 1)
        return Raw.any(self, reductionIndices: axes).scalarized()
    }

    /// Performs a logical AND operation along the specified axes. The reduced dimensions are
    /// removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    func all(squeezingAxes axes: Int...) -> Tensor {
        let axes = axes.map(Int32.init)
        return Raw.all(self, reductionIndices: Tensor<Int32>(axes), keepDims: false)
    }

    /// Performs a logical AND operation along the specified axes. The reduced dimensions are
    /// removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    func any(squeezingAxes axes: Int...) -> Tensor {
        let axes = axes.map(Int32.init)
        return Raw.any(self, reductionIndices: Tensor<Int32>(axes), keepDims: false)
    }

    /// Performs a logical AND operation along the specified axes. The reduced dimensions are
    /// retained with value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    func all(alongAxes axes: Int...) -> Tensor {
        let axes = axes.map(Int32.init)
        return Raw.all(self, reductionIndices: Tensor<Int32>(axes), keepDims: true)
    }

    /// Performs a logical OR operation along the specified axes. The reduced
    /// dimensions are retained with value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    func any(alongAxes axes: Int...) -> Tensor {
        let axes = axes.map(Int32.init)
        return Raw.any(self, reductionIndices: Tensor<Int32>(axes), keepDims: true)
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
        return Raw.max(self, reductionIndices: axes, keepDims: false)
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
        return max(squeezingAxes: axes)
    }

    /// Returns the minimum values along the specified axes. The reduced dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(
        wrt: self,
        vjp: _vjpMinOrMax(squeezingAxes:) where Scalar: TensorFlowFloatingPoint)
    func min(squeezingAxes axes: Tensor<Int32>) -> Tensor {
        return Raw.min(self, reductionIndices: axes, keepDims: false)
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
        return min(squeezingAxes: axes)
    }

    /// Returns the indices of the maximum values along the specified axes. The reduced dimensions
    /// are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    func argmax(squeezingAxis axis: Int) -> Tensor<Int32> {
        return Raw.argMax(self, dimension: Tensor<Int32>(Int32(axis)))
    }

    /// Returns the indices of the minimum values along the specified axes. The reduced dimensions
    /// are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    func argmin(squeezingAxis axis: Int) -> Tensor<Int32> {
        return Raw.argMin(self, dimension: Tensor<Int32>(Int32(axis)))
    }

    /// Returns the minimum along the specified axes. The reduced dimensions are retained with
    /// value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpMinOrMax(alongAxes:) where Scalar: TensorFlowFloatingPoint)
    func min(alongAxes axes: Tensor<Int32>) -> Tensor {
        return Raw.min(self, reductionIndices: axes, keepDims: true)
    }

    /// Returns the minimum along the specified axes. The reduced dimensions are retained with
    /// value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    func min(alongAxes axes: [Int]) -> Tensor {
        let axes = axes.map(Int32.init)
        return min(alongAxes: Tensor<Int32>(axes))
    }

    /// Returns the minimum along the specified axes. The reduced dimensions are retained with
    /// value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    func min(alongAxes axes: Int...) -> Tensor {
        return min(alongAxes: axes)
    }

    /// Returns the minimum along the specified axes. The reduced dimensions are retained with
    /// value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpMinOrMax(alongAxes:) where Scalar: TensorFlowFloatingPoint)
    func max(alongAxes axes: Tensor<Int32>) -> Tensor {
        return Raw.max(self, reductionIndices: axes, keepDims: true)
    }

    /// Returns the minimum along the specified axes. The reduced dimensions are retained with
    /// value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    func max(alongAxes axes: [Int]) -> Tensor {
        let axes = axes.map(Int32.init)
        return max(alongAxes: Tensor<Int32>(axes))
    }

    /// Returns the minimum along the specified axes. The reduced dimensions are retained with
    /// value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    func max(alongAxes axes: Int...) -> Tensor {
        return max(alongAxes: axes)
    }

    /// Returns the index of the maximum value of the flattened scalars.
    @inlinable
    func argmax() -> Tensor<Int32> {
        return flattened().argmax(squeezingAxis: 0)
    }

    /// Returns the index of the minimum value of the flattened scalars.
    @inlinable
    func argmin() -> Tensor<Int32> {
        return flattened().argmin(squeezingAxis: 0)
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
        return Raw.sum(self, reductionIndices: axes, keepDims: false)
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
        return sum(squeezingAxes: axes)
    }

    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func sum() -> Tensor {
        return flattened().sum(squeezingAxes: 0)
    }

    /// Returns the sum along the specified axes. The reduced dimensions are retained with value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpSum(alongAxes:) where Scalar: TensorFlowFloatingPoint)
    func sum(alongAxes axes: Tensor<Int32>) -> Tensor {
        return Raw.sum(self, reductionIndices: axes, keepDims: true)
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
        return sum(alongAxes: axes)
    }

    // MARK: - Product

    /// Returns the product along the specified axes. The reduced dimensions are removed.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
    // TODO: Make this @differentiable.
    @inlinable
    func product(squeezingAxes axes: Tensor<Int32>) -> Tensor {
        return Raw.prod(self, reductionIndices: axes, keepDims: false)
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
        return product(squeezingAxes: axes)
    }

    @inlinable
    func product() -> Tensor {
        return flattened().product(squeezingAxes: 0)
    }

    /// Returns the product along the specified axes. The reduced dimensions are retained with
    /// value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    func product(alongAxes axes: Tensor<Int32>) -> Tensor {
        return Raw.prod(self, reductionIndices: axes, keepDims: true)
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
        return product(alongAxes: axes)
    }

    // MARK: - Mean

    /// Returns the arithmetic mean along the specified axes. The reduced dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpMean(squeezingAxes:) where Scalar: TensorFlowFloatingPoint)
    func mean(squeezingAxes axes: Tensor<Int32>) -> Tensor {
        return Raw.mean(self, reductionIndices: axes, keepDims: false)
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
        return mean(squeezingAxes: axes)
    }

    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func mean() -> Tensor {
        return flattened().mean(squeezingAxes: [0])
    }

    /// Returns the arithmetic mean along the specified axes. The reduced dimensions are retained
    /// with value 1.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpMean(alongAxes:) where Scalar: TensorFlowFloatingPoint)
    func mean(alongAxes axes: Tensor<Int32>) -> Tensor {
        return Raw.mean(self, reductionIndices: axes, keepDims: true)
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
        return mean(alongAxes: axes)
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
        return variance(squeezingAxes: axes)
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
        return variance(alongAxes: axes)
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
        let count = Raw.gather(params: shapeTensor, indices: axes).product()
        return (value, { [shape = shapeTensor] in $0.broadcasted(toShape: shape) / Tensor(count) })
    }

    @inlinable
    func _vjpMean(squeezingAxes axes: Tensor<Int32>) -> (Tensor, (Tensor) -> Tensor) {
        let value = mean(squeezingAxes: axes)
        let count = Raw.gather(params: shapeTensor, indices: axes).product()
        return (value, { [shape = shapeTensor] v in
	      let unsqueezed = v.expandingShape(at: axes.scalars.map { Int($0) })
	      return unsqueezed.broadcasted(toShape: shape) / Tensor(count)
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
        return sqrt(variance(squeezingAxes: axes))
    }

    /// Returns the standard deviation of the elements along the specified axes. The reduced
    /// dimensions are retained with value `1`. Does not apply Bessel's correction.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self)
    func standardDeviation(squeezingAxes axes: [Int]) -> Tensor {
        return sqrt(variance(squeezingAxes: axes))
    }

    /// Returns the standard deviation of the elements along the specified axes. The reduced
    /// dimensions are retained with value `1`. Does not apply Bessel's correction.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self)
    func standardDeviation(squeezingAxes axes: Int...) -> Tensor {
        return standardDeviation(squeezingAxes: axes)
    }

    /// Returns the standard deviation of the elements along the specified axes. The reduced
    /// dimensions are retained with value `1`. Does not apply Bessel's correction.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self)
    func standardDeviation() -> Tensor {
        // Reduce along all dimensions.
        return standardDeviation(squeezingAxes: Array(0..<shape.rank))
    }

    /// Returns the standard deviation of the elements along the specified axes. The reduced
    /// dimensions are retained with value `1`. Does not apply Bessel's correction.
    ///
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(wrt: self)
    func standardDeviation(alongAxes axes: Tensor<Int32>) -> Tensor {
        return sqrt(variance(alongAxes: axes))
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
        return sqrt(variance(alongAxes: axes))
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
    transposed transposeA: Bool = false,
    _ rhs: Tensor<Scalar>,
    transposed transposeB: Bool = false
) -> Tensor<Scalar> {
    switch (lhs.rank, rhs.rank) {
    case (3..., 3...):
        return Raw.batchMatMulV2(lhs, rhs, adjX: transposeA, adjY: transposeB)
    case (2, 3...):
        return Raw.batchMatMulV2(lhs.expandingShape(at: 1), rhs, adjX: transposeA, adjY: transposeB)
    case (3..., 2):
        return Raw.batchMatMulV2(lhs, rhs.expandingShape(at: 1), adjX: transposeA, adjY: transposeB)
    default:
        return Raw.matMul(lhs, rhs, transposeA: transposeA, transposeB: transposeB)
    }
}

@inlinable
internal func _vjpMatmul<Scalar: TensorFlowFloatingPoint>(
    _ lhs: Tensor<Scalar>,
    transposed transposeA: Bool = false,
    _ rhs: Tensor<Scalar>,
    transposed transposeB: Bool = false
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
    let value = matmul(lhs, transposed: transposeA, rhs, transposed: transposeB)
    return (value, { v in
        let (lhsGrad, rhsGrad): (Tensor<Scalar>, Tensor<Scalar>)
        switch (transposeA, transposeB) {
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
        switch (lhs.rank, rhs.rank) {
        case (3..., 3...): return (lhsGrad.sum(squeezingAxes: 1), rhsGrad)
        case (3..., 2): return (lhsGrad, rhsGrad.sum(squeezingAxes: 1))
        default: return (lhsGrad, rhsGrad)
        }
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
        return matmul(lhs, rhs)
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
        return _vjpMatmul(lhs, rhs)
    }
}
