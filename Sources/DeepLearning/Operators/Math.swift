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
// Universal Functions
//===------------------------------------------------------------------------------------------===//

/// Returns the values of the specified tensor rounded to the nearest integer, element-wise.
@inlinable
@differentiable(vjp: _vjpRound)
public func round<T : TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.round(x)
}

@inlinable
internal func _vjpRound<T : TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    return (round(x), { v in Tensor<T>(zerosLike: v) })
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

// /// Computes the log-sigmoid of the specified tensor element-wise. Specifically, 
// /// `y = log(1 / (1 + exp(-x)))`. For numerical stability, we use `y = -softplus(-x)`.
// @inlinable
// @differentiable
// public func logSigmoid<T : TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
//     return -softplus(-x)
// }

// /// Computes the softplus function for the specified tensor element-wise. The softplus function is 
// /// defined as `log(exp(x) + 1)`.
// @inlinable
// @differentiable(vjp: _vjpSoftplus)
// public func softplus<T : TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
//     return Raw.softplus(features: x)
// }

// @inlinable
// internal func _vjpSoftplus<T : TensorFlowFloatingPoint>(
//     _ x: Tensor<T>
// ) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
//     return (softplus(x), { v in v * sigmoid(x) })
// }


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
