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

@_exported import TensorFlow

/// A neural network layer.
///
/// Types that conform to `Module` represent functions that map inputs to
/// outputs. They may have an internal state represented by parameters, such as
/// weight tensors.
///
/// `Layer` instances define a differentiable `applied(to:)` method for mapping
/// inputs to outputs.
public protocol Layer: Differentiable & KeyPathIterable
    where AllDifferentiableVariables: KeyPathIterable {
    /// The input type of the layer.
    associatedtype Input: TensorGroup & Differentiable
    /// The output type of the layer.
    associatedtype Output: TensorGroup & Differentiable

    /// Returns the output obtained from applying to an input.
    @differentiable(wrt: (self, .0))
    func applied(to input: Input) -> Output
}

public extension Layer {
    func valueWithPullback(at input: Input)
        -> (output: Output,
            pullback: (Output.CotangentVector)
                -> (layerGradient: CotangentVector, inputGradient: Input.CotangentVector)) {
        let (out, pullback) = _valueWithPullback(at: self, input, in: Self.applied(to:))
        return (out, pullback)
    }
}

// FIXME(SR-9666): @differentiable attribute deserialization failure.
// public struct Dense<Scalar>: VectorNumeric, Layer
//     where Scalar : FloatingPoint & Differentiable & TensorFlowScalar {
//     public var weight: Tensor<Scalar>
//     public var bias: Tensor<Scalar>
//
//     @differentiable(wrt: (self, .0), vjp: _vjpApplied(to:))
//     public func applied(to input: Tensor<Scalar>) -> Tensor<Scalar> {
//         return matmul(input, weight) + bias
//     }
//
//     @usableFromInline
//     func _vjpApplied(to input: Tensor<Scalar>)
//         -> (Tensor<Scalar>, (Tensor<Scalar>) -> (Dense, Tensor<Scalar>)) {
//       let r0 = matmul(input, weight)
//       let r1 = r0 + bias
//       func pullback(_ v: Tensor<Scalar>) -> (Dense, Tensor<Scalar>) {
//         return (Dense(weight: matmul(input.transposed(), v), bias: v),
//                 matmul(v, weight.transposed()))
//       }
//       return (r1, pullback)
//     }
// }
