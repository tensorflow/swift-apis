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

/// Computes the mean squared error between logits and labels.
///
/// - Parameters:
///   - logits: One-hot encoded outputs from a neural network.
///   - labels: One-hot encoded values that correspond to the correct output.
@differentiable
public func meanSquaredError<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>, expected: Tensor<Scalar>) -> Tensor<Scalar> {
    return (expected - predicted).squared().mean()
}

/// Computes the softmax cross entropy (categorical cross entropy) between logits and labels.
///
/// - Parameters:
///   - logits: One-hot encoded outputs from a neural network.
///   - labels: One-hot encoded values that correspond to the correct output.
@differentiable
public func softmaxCrossEntropy<Scalar: TensorFlowFloatingPoint>(
    logits: Tensor<Scalar>, labels: Tensor<Scalar>) -> Tensor<Scalar> {
    return -(labels * logSoftmax(logits)).mean(alongAxes: 0).sum()
}

/// Computes the sigmoid cross entropy (binary cross entropy) between logits and labels.
///
/// - Parameters:
///   - logits: Single continuous values from `0` to `1`.
///   - labels: Integer values that correspond to the correct output.
@differentiable
public func sigmoidCrossEntropy<Scalar: TensorFlowFloatingPoint>(
    logits: Tensor<Scalar>, labels: Tensor<Scalar>
) -> Tensor<Scalar> {
    let loss = labels * log(logits) +
        (Tensor<Scalar>(ones: labels.shape) - labels) *
        log(Tensor<Scalar>(ones: logits.shape) - logits)
    return -loss.mean(alongAxes: 0).sum()
}
