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
public func round<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    return Raw.round(x)
}

@inlinable
internal func _vjpRound<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
  return (round(x), { v in Tensor<T>(zerosLike: v) })
}

/// Computes the sigmoid of the specified tensor element-wise.
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

/// Computes the log-sigmoid of the specified tensor element-wise. Specifically, 
/// `y = log(1 / (1 + exp(-x)))`. For numerical stability, we use `y = -softplus(-x)`.
@inlinable
@differentiable
public func logSigmoid<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  return -softplus(-x)
}

/// Computes the softplus function for the specified tensor element-wise. The softplus function is 
/// defined as `log(exp(x) + 1)`.
@inlinable
@differentiable(vjp: _vjpSoftplus)
public func softplus<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  return Raw.softplus(features: x)
}

@inlinable
internal func _vjpSoftplus<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
  return (softplus(x), { v in v * sigmoid(x) })
}
