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
public protocol Layer: Differentiable
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

// public struct Dense: Layer {
//     // FIXME(SR-9657): TBDGen does not handle @differentiable vars yet.
//     public var weight: Tensor<Float>
//     public var bias: Tensor<Float>
//
//     // FIXME(SR-9658): Functions that implement @differentiable protocol requirements should be
//     // enforced to write the attribute
//     @differentiable(wrt: (self, .0))
//     public func applied(to input: Tensor<Float>) -> Tensor<Float> {
//         return matmul(input, weight) + bias
//     }
// }
