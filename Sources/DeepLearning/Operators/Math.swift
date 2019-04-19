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

/// Computes the log-sigmoid of the specified tensor element-wise. Specifically, 
/// `y = log(1 / (1 + exp(-x)))`. For numerical stability, we use `y = -softplus(-x)`.
@inlinable
@differentiable
public func logSigmoid<T : TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return -softplus(-x)
}

/// Computes the softplus function for the specified tensor element-wise. The softplus function is 
/// defined as `log(exp(x) + 1)`.
@inlinable
@differentiable(vjp: _vjpSoftplus)
public func softplus<T : TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.softplus(features: x)
}

@inlinable
internal func _vjpSoftplus<T : TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    return (softplus(x), { v in v * sigmoid(x) })
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
public func softmax<T : TensorFlowFloatingPoint>(
    _ x: Tensor<T>,
    alongAxis axis: Int
) -> Tensor<T> {
    let expx = exp(x)
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
