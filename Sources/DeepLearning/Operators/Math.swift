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

#if !COMPILING_TENSORFLOW_MODULE
import TensorFlow
#endif

#if COMPILING_TENSORFLOW_MODULE
infix operator .> : ComparisonPrecedence
#endif

// TODO:
// - Consider explicit broadcasting for elementwise binary ops when
//   scalarization and rank getter are implemented.

//===------------------------------------------------------------------------------------------===//
// Additive Group
//===------------------------------------------------------------------------------------------===//

extension Tensor : AdditiveArithmetic where Scalar : Numeric {
    /// A scalar zero tensor.
    @inlinable
    public static var zero: Tensor {
        get {
        return Tensor(zeros: [])
        }
    }

    /// Adds two tensors and produces their sum.
    /// - Note: `+` supports broadcasting.
    @inlinable
    @differentiable(vjp: _vjpAdd(lhs:rhs:) where Scalar : TensorFlowFloatingPoint)
    public static func + (lhs: Tensor, rhs: Tensor) -> Tensor {
        return Raw.add(lhs, rhs)
    }

    /// Subtracts one tensor from another and produces their difference.
    /// - Note: `-` supports broadcasting.
    @inlinable
    @differentiable(vjp: _vjpSubtract(lhs:rhs:) where Scalar : TensorFlowFloatingPoint)
    public static func - (lhs: Tensor, rhs: Tensor) -> Tensor {
        return Raw.sub(lhs, rhs)
    }
}

internal extension Tensor where Scalar : TensorFlowFloatingPoint {
    @inlinable
    static func _vjpAdd(lhs: Tensor, rhs: Tensor) -> (Tensor, (Tensor) -> (Tensor, Tensor)) {
        return (lhs + rhs, { [lhsShape = lhs.shapeTensor, rhsShape = rhs.shapeTensor] v in
            (v.unbroadcast(toShape: lhsShape), v.unbroadcast(toShape: rhsShape))
        })
    }

    @inlinable
    static func _vjpSubtract(lhs: Tensor, rhs: Tensor) -> (Tensor, (Tensor) -> (Tensor, Tensor)) {
        return (lhs - rhs, { [lhsShape = lhs.shapeTensor, rhsShape = rhs.shapeTensor] v in
            (v.unbroadcast(toShape: lhsShape), -v.unbroadcast(toShape: rhsShape))
        })
    }
}

//===------------------------------------------------------------------------------------------===//
// Vector Space
//===------------------------------------------------------------------------------------------===//

extension Tensor : VectorNumeric where Scalar : Numeric {
    /// Multiplies the scalar with every scalar of the tensor and produces the product.
    @inlinable
    @differentiable(vjp: _vjpMultiply(lhs:rhs:) where Scalar : TensorFlowFloatingPoint)
    public static func * (lhs: Scalar, rhs: Tensor) -> Tensor {
        return Tensor(lhs) * rhs
    }
}

internal extension Tensor where Scalar : TensorFlowFloatingPoint {
    @inlinable
    static func _vjpMultiply(lhs: Tensor, rhs: Tensor) -> (Tensor, (Tensor) -> (Tensor, Tensor)) {
        return (lhs * rhs, { [lhsShape = lhs.shapeTensor, rhsShape = rhs.shapeTensor] v in
            ((rhs * v).unbroadcast(toShape: lhsShape), (lhs * v).unbroadcast(toShape: rhsShape))
        })
    }
}

extension Tensor : ShapedVectorNumeric where Scalar : Numeric {}

extension Tensor : Differentiable where Scalar : TensorFlowFloatingPoint {
    public typealias TangentVector = Tensor
    public typealias CotangentVector = Tensor
    public typealias AllDifferentiableVariables = Tensor

    @inlinable
    public func tangentVector(from cotangent: CotangentVector) -> TangentVector {
        return cotangent
    }
}

//===------------------------------------------------------------------------------------------===//
// Additional Element-wise Operators
//===------------------------------------------------------------------------------------------===//

public extension Tensor where Scalar : Numeric {
    /// Adds the scalar to every scalar of the tensor and produces the sum.
    @inlinable
    @differentiable(vjp: _vjpAdd(lhs:rhs:) where Scalar : TensorFlowFloatingPoint)
    static func + (lhs: Scalar, rhs: Tensor) -> Tensor {
        return Tensor(lhs) + rhs
    }

    /// Adds the scalar to every scalar of the tensor and produces the sum.
    @inlinable
    @differentiable(vjp: _vjpAdd(lhs:rhs:) where Scalar : TensorFlowFloatingPoint)
    static func + (lhs: Tensor, rhs: Scalar) -> Tensor {
        return lhs + Tensor(rhs)
    }

    /// Subtracts the scalar from every scalar of the tensor and produces the difference.
    @inlinable
    @differentiable(vjp: _vjpSubtract(lhs:rhs:) where Scalar : TensorFlowFloatingPoint)
    static func - (lhs: Scalar, rhs: Tensor) -> Tensor {
        return Tensor(lhs) - rhs
    }

    /// Subtracts the scalar from every scalar of the tensor and produces the difference
    @inlinable
    @differentiable(vjp: _vjpSubtract(lhs:rhs:) where Scalar : TensorFlowFloatingPoint)
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
    @differentiable(vjp: _vjpMultiply(lhs:rhs:) where Scalar : TensorFlowFloatingPoint)
    static func * (lhs: Tensor, rhs: Tensor) -> Tensor {
        return Raw.mul(lhs, rhs)
    }

    /// Multiplies the scalar with every scalar of the tensor and produces the product.
    @inlinable
    @differentiable(vjp: _vjpMultiply(lhs:rhs:) where Scalar : TensorFlowFloatingPoint)
    static func * (lhs: Tensor, rhs: Scalar) -> Tensor {
        return lhs * Tensor(rhs)
    }

    /// Multiplies two tensors and stores the result in the left-hand-side variable.
    /// - Note: `*=` supports broadcasting.
    @inlinable
    static func *= (lhs: inout Tensor, rhs: Tensor) {
        lhs = lhs * rhs
    }

    @inlinable
    static func *= (lhs: inout Tensor, rhs: Scalar) {
        lhs = lhs * rhs
    }

    /// Returns the quotient of dividing the first tensor by the second.
    /// - Note: `/` supports broadcasting.
    @inlinable
    @differentiable(vjp: _vjpDivide(lhs:rhs:) where Scalar : TensorFlowFloatingPoint)
    static func / (lhs: Tensor, rhs: Tensor) -> Tensor {
        return Raw.div(lhs, rhs)
    }

    /// Returns the quotient of dividing the scalar by the tensor, broadcasting the scalar.
    @inlinable
    @differentiable(vjp: _vjpDivide(lhs:rhs:) where Scalar : TensorFlowFloatingPoint)
    static func / (lhs: Scalar, rhs: Tensor) -> Tensor {
        return Tensor(lhs) / rhs
    }

    /// Returns the quotient of dividing the tensor by the scalar, broadcasting the scalar.
    @inlinable
    @differentiable(vjp: _vjpDivide(lhs:rhs:) where Scalar : TensorFlowFloatingPoint)
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

internal extension Tensor where Scalar : TensorFlowFloatingPoint {
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
        return (lhs - rhs, { v in (v, 0 - v.sum().scalarized()) })
    }

    @inlinable
    static func _vjpSubtract(lhs: Scalar, rhs: Tensor) -> (Tensor, (Tensor) -> (Scalar, Tensor)) {
        return (lhs - rhs, { v in (v.sum().scalarized(), 0 - v) })
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
            ((v / rhs).unbroadcast(toShape: lhsShape),
             ((-lhs) / rhs.squared() * v).unbroadcast(toShape: rhsShape))
        })
    }

    @inlinable
    static func _vjpDivide(lhs: Tensor, rhs: Scalar) -> (Tensor, (Tensor) -> (Tensor, Scalar)) {
        return (lhs / rhs, { v in 
            (v / rhs, (v * (0 - lhs) / Tensor(rhs).squared()).sum().scalarized())
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

public extension Tensor where Scalar : SignedNumeric {
    /// Computes the negation of the specified tensor element-wise.
    @inlinable
    @differentiable(vjp: _vjpNegate(_:) where Scalar : TensorFlowFloatingPoint)
    static prefix func - (rhs: Tensor) -> Tensor {
        return Raw.neg(rhs)
    }
}

internal extension Tensor where Scalar : TensorFlowFloatingPoint {
    @inlinable
    static func _vjpNegate(_ x: Tensor) -> (Tensor, (Tensor) -> Tensor) {
        return (-x, { v in -v })
    }
}

/// Computes the absolute value of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpAbs(_:) where T : TensorFlowFloatingPoint)
public func abs<T : SignedNumeric>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.abs(x)
}

@inlinable
internal func _vjpAbs<T : TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let sign = Raw.sign(x)
    return (abs(x), { v in v * sign })
}

/// Computes the natural logarithm of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpLog(_:) where T : TensorFlowFloatingPoint)
public func log<T : FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.log(x)
}

@inlinable
internal func _vjpLog<T : TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    return (log(x), { v in v / x })
}

/// Computes `sin` of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpSin(_:) where T : TensorFlowFloatingPoint)
public func sin<T : FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.sin(x)
}

@inlinable
internal func _vjpSin<T : TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    return (sin(x), { v in v * cos(x) })
}

/// Computes `cos` of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpCos(_:) where T : TensorFlowFloatingPoint)
public func cos<T : FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.cos(x)
}

@inlinable
internal func _vjpCos<T : TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    return (cos(x), { v in -v * sin(x) })
}

/// Computes `tan` of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpTan(_:) where T : TensorFlowFloatingPoint)
public func tan<T : FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.tan(x)
}

@inlinable
internal func _vjpTan<T : TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let value = tan(x)
    return (value, { v in v * (1 + value.squared()) })
}

/// Computes `sinh` of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpSinh(_:) where T : TensorFlowFloatingPoint)
public func sinh<T : FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.sinh(x)
}

@inlinable
internal func _vjpSinh<T : TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    return (sinh(x), { v in v * cosh(x) })
}

/// Computes `cosh` of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpCosh(_:) where T : TensorFlowFloatingPoint)
public func cosh<T : FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.cosh(x)
}

@inlinable
internal func _vjpCosh<T : TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    return (cosh(x), { v in v * sinh(x) })
}

/// Computes `tanh` of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpTanh(_:) where T : TensorFlowFloatingPoint)
public func tanh<T : FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.tanh(x)
}

@inlinable
internal func _vjpTanh<T : TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let value = tanh(x)
    return (value, { v in v * (1 - value.squared()) })
}

/// Computes the square of the tensor.
public extension Tensor where Scalar : Numeric {
    @inlinable
    @differentiable(wrt: self, vjp: _vjpSquared() where Scalar : TensorFlowFloatingPoint)
    func squared() -> Tensor {
        return Raw.square(self)
    }
}

internal extension Tensor where Scalar : TensorFlowFloatingPoint {
    @inlinable
    func _vjpSquared() -> (Tensor, (Tensor) -> Tensor) {
        return (squared(), { 2 * self * $0 })
    }
}

/// Computes the square root of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpSqrt(_:) where T : TensorFlowFloatingPoint)
public func sqrt<T : FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.sqrt(x)
}

@inlinable
internal func _vjpSqrt<T : TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let value = sqrt(x)
    return (value, { v in v / (2 * value) })
}

/// Computes the inverse square root of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpRsqrt(_:) where T : TensorFlowFloatingPoint)
public func rsqrt<T : FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.rsqrt(x)
}

@inlinable
internal func _vjpRsqrt<T : TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let value = rsqrt(x)
    return (value, { v in -v / 2 * value })
}

/// Computes `exp` of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpExp(_:) where T : TensorFlowFloatingPoint)
public func exp<T : FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.exp(x)
}

@inlinable
internal func _vjpExp<T : TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let value = exp(x)
    return (value, { v in value * v })
}

/// Computes the ceiling of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpCeil(_:) where T : TensorFlowFloatingPoint)
public func ceil<T : FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.ceil(x)
}

@inlinable
internal func _vjpCeil<T : TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    return (ceil(x), { _ in Tensor(0).broadcast(like: x) })
}

/// Computes the floor of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpFloor(_:) where T : TensorFlowFloatingPoint)
public func floor<T : FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.floor(x)
}

@inlinable
internal func _vjpFloor<T : TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    return (floor(x), { _ in Tensor(0).broadcast(like: x) })
}

//===------------------------------------------------------------------------------------------===//
// Element-wise Binary Math Functions
//===------------------------------------------------------------------------------------------===//

/// Computes the power of the first tensor to the second tensor.
@inlinable
@differentiable(vjp: _vjpPow(_:_:) where T : TensorFlowFloatingPoint)
public func pow<T>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> where T : FloatingPoint {
    return Raw.pow(lhs, rhs)
}

@inlinable
internal func _vjpPow<T : TensorFlowFloatingPoint>(
    _ x: Tensor<T>, _ y: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> (Tensor<T>, Tensor<T>)) {
    let value = pow(x, y)
    return (value, { v in
        ((v * y * pow(x, y-1)).unbroadcast(like: x),
        (v * log(x) * value).unbroadcast(like: y))
    })
}

/// Computes the power of the scalar to the tensor, broadcasting the scalar.
@inlinable
// @differentiable(where T : TensorFlowFloatingPoint)
public func pow<T>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T> where T : FloatingPoint {
    return pow(Tensor(lhs), rhs)
}

/// Computes the power of the tensor to the scalar, broadcasting the scalar.
@inlinable
// @differentiable(where T : TensorFlowFloatingPoint)
public func pow<T>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T> where T : FloatingPoint {
    return pow(lhs, Tensor(rhs))
}

/// Computes the element-wise maximum of two tensors.
/// - Note: `max` supports broadcasting.
@inlinable
@differentiable(vjp: _vjpMax(_:_:) where T : TensorFlowFloatingPoint)
public func max<T>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> where T : Numeric & Comparable {
    return Raw.maximum(lhs, rhs)
}

@inlinable
internal func _vjpMax<T : TensorFlowFloatingPoint>(
    _ x: Tensor<T>, _ y: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> (Tensor<T>, Tensor<T>)) {
    let value = max(x, y)
    return (value, { v in _vjpMinMaxHelper(x, y, originalValue: value, vector: v) })
}

/// Computes the element-wise maximum of the scalar and the tensor, broadcasting the scalar.
@inlinable
// @differentiable(where T : TensorFlowFloatingPoint)
public func max<T>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T> where T : Numeric & Comparable {
    return max(Tensor(lhs), rhs)
}

/// Computes the element-wise maximum of the scalar and the tensor, broadcasting the scalar.
@inlinable
// @differentiable(where T : TensorFlowFloatingPoint)
public func max<T>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T> where T : Numeric & Comparable {
    return max(lhs, Tensor(rhs))
}

/// Computes the element-wise minimum of two tensors.
/// - Note: `min` supports broadcasting.
@inlinable
@differentiable(vjp: _vjpMin(_:_:) where T : TensorFlowFloatingPoint)
public func min<T>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> where T : Numeric & Comparable {
    return Raw.minimum(lhs, rhs)
}

@inlinable
internal func _vjpMin<T : TensorFlowFloatingPoint>(
    _ x: Tensor<T>, _ y: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> (Tensor<T>, Tensor<T>)) {
    let value = min(x, y)
    return (value, { v in _vjpMinMaxHelper(x, y, originalValue: value, vector: v) })
}

/// Computes the element-wise minimum of the scalar and the tensor, broadcasting the scalar.
@inlinable
// @differentiable(where T : TensorFlowFloatingPoint)
public func min<T>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T> where T : Numeric & Comparable {
    return min(Tensor(lhs), rhs)
}

/// Computes the element-wise minimum of the scalar and the tensor, broadcasting the scalar.
@inlinable
// @differentiable(where T : TensorFlowFloatingPoint)
public func min<T>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T> where T : Numeric & Comparable {
    return min(lhs, Tensor(rhs))
}

@inlinable
internal func _vjpMinMaxHelper<T : TensorFlowFloatingPoint>(
    _ x: Tensor<T>,
    _ y: Tensor<T>,
    originalValue: Tensor<T>,
    vector: Tensor<T>
) -> (Tensor<T>, Tensor<T>) {
    let denom = 1 + Tensor<T>(x .== y)
    let dfdx = vector * Tensor<T>(x .== originalValue) / denom
    let dfdy = vector * Tensor<T>(y .== originalValue) / denom
    return (dfdx.unbroadcast(like: x), dfdy.unbroadcast(like: y))
}

//===------------------------------------------------------------------------------------------===//
// Selection Functions
//===------------------------------------------------------------------------------------------===//

public extension Tensor where Scalar == Bool {
    /// Returns a new tensor containing elements from either `left` or `right`,
    /// depending on the elements of `self`.
    ///
    /// `self` acts as a mask that chooses, based on the value at each scalar,
    ///  whether the corresponding scalar in the output should be taken from
    /// `left` (if `true`) or `right` (if `false`).
    ///
    /// - Precondition: `left` and `right` must have the same shape. If
    ///   `left` and `right` are scalar, then `self` must also be scalar. If
    ///   `left` and `right` have rank greater than or equal to 1, then `self`
    ///   must be either have the same shape as `left` or be a 1-D `Tensor` such
    ///   that `self.scalarCount == left[0]`.
    @available(*, deprecated, message: "Use '.replacing(with:mask:)' instead")
    @inlinable
    func selecting<T>(_ left: Tensor<T>, _ right: Tensor<T>) -> Tensor<T> {
        return left.replacing(with: right, where: self)
    }
}

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
    @differentiable(wrt: (self, other), vjp: _vjpReplacing where Scalar : TensorFlowFloatingPoint)
    func replacing(with other: Tensor, where mask: Tensor<Bool>) -> Tensor {
        return Raw.select(condition: mask, t: self, e: other)
    }
}

internal extension Tensor where Scalar : TensorFlowFloatingPoint {
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

/// Computes the sigmoid of the specified tensor element-wise.
/// Specifically, computes `1 / (1 + exp(-x))`.
@inlinable
@differentiable(vjp: _vjpSigmoid)
public func sigmoid<T : TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.sigmoid(x)
}

@inlinable
internal func _vjpSigmoid<T : TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    return (sigmoid(x), { v in Raw.sigmoidGrad(x, dy: v) })
}

/// Computes the softmax of the specified tensor along the last axis.
/// Specifically, computes `exp(x) / exp(x).sum(alongAxes: -1)`.
@inlinable
@differentiable(vjp: _vjpSoftmax(_:) where T : TensorFlowFloatingPoint)
public func softmax<T : FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.softmax(logits: x)
}

/// Computes the softmax of the specified tensor along the specified axis.
/// Specifically, computes `exp(x) / exp(x).sum(alongAxes: axis)`.
@inlinable
// TODO: [AD].
public func softmax<T : TensorFlowFloatingPoint>(
    _ x: Tensor<T>,
    alongAxis axis: Int
) -> Tensor<T> {
    let expx = exp(x)
    // TODO: [BUG] keepDims = true for the sum.
    return expx / expx.sum(alongAxes: axis)
}

@inlinable
func _vjpSoftmax<T : TensorFlowFloatingPoint>(
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
@differentiable(vjp: _vjpLogSoftmax(_:) where T : TensorFlowFloatingPoint)
public func logSoftmax<T : FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.logSoftmax(logits: x)
}

@inlinable
func _vjpLogSoftmax<T : TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
  let value = logSoftmax(x)
  return (value, { v in
    v - v.sum(alongAxes: -1) * exp(value)
  })
}

/// Computes `relu` of the specified tensor element-wise.
/// Specifically, computes `max(0, x)`.
@inlinable
@differentiable(vjp: _vjpRelu(_:) where T : TensorFlowFloatingPoint)
public func relu<T : FloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return max(0, x)
}

@inlinable
func _vjpRelu<T : TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    return (relu(x), { v in Tensor(x .> 0) * v })
}
