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

/// A neural network layer.
///
/// Types that conform to `Layer` represent functions that map inputs to outputs. They may have an
/// internal state represented by parameters, such as weight tensors.
///
/// `Layer` instances define a differentiable `applied(to:)` method for mapping inputs to
/// outputs.
public protocol Layer: Differentiable & KeyPathIterable
    where AllDifferentiableVariables: KeyPathIterable {
    /// The input type of the layer.
    associatedtype Input: Differentiable
    /// The output type of the layer.
    associatedtype Output: Differentiable

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    func callAsFunction(_ input: Input) -> Output
}

public extension Layer {
    @differentiable
    func call(_  input: Input) -> Output {
        return callAsFunction(input)
    }
}

public extension Layer {
    /// Returns the inference output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The inference output.
    @differentiable
    func inferring(from input: Input) -> Output {
        return withLearningPhase(LearningPhase.inference) { self(input) }
    }

    // TODO(rxwei): Remove this custom VJP once differentiation supports currying.
    @differentiating(inferring(from:))
    @usableFromInline
    internal func _vjpInferring(from input: Input)
        -> (value: Output, pullback: (Output.TangentVector)
            -> (TangentVector, Input.TangentVector)) {
        return withLearningPhase(LearningPhase.inference) {
            let (output, pullback) = appliedForBackpropagation(to: input)
            return (output, { v in pullback(v) })
        }
    }

    typealias Backpropagator = (_ direction: Output.TangentVector)
        -> (layerGradient: TangentVector, inputGradient: Input.TangentVector)

    /// Returns the inference output and the backpropagation function obtained from applying the
    /// layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: A tuple containing the output and the backpropagation function. The
    ///   backpropagation function (a.k.a. backpropagator) takes a direction vector and returns the
    ///   gradients at the layer and at the input, respectively.
    func appliedForBackpropagation(to input: Input)
        -> (output: Output, backpropagator: Backpropagator) {
        let (out, pullback) = valueWithPullback(at: input) { layer, input in
            return layer(input)
        }
        return (out, pullback)
    }
}

public extension Differentiable {
    /// Returns the output computed by applying a sequence of layers to the previous layer's output,
    /// except that the first layer's input is `self`.
    ///
    /// - Parameters:
    ///   - l1: The first layer.
    ///   - l2: The second layer.
    /// - Returns: The final layer's output after sequential application.
    @differentiable
    func sequenced<L1: Layer, L2: Layer>(through l1: L1, _ l2: L2) -> L2.Output
        where L1.Input == Self, L1.Output == L2.Input {
        let o1 = l1(self)
        return l2(o1)
    }

    /// Returns the output computed by applying a sequence of layers to the previous layer's output,
    /// except that the first layer's input is `self`.
    ///
    /// - Parameters:
    ///   - l1: The first layer.
    ///   - l2: The second layer.
    ///   - l3: The third layer.
    /// - Returns: The final layer's output after sequential application.
    @differentiable
    func sequenced<L1: Layer, L2: Layer, L3: Layer>(through l1: L1, _ l2: L2, _ l3: L3) -> L3.Output
        where L1.Input == Self, L1.Output == L2.Input, L2.Output == L3.Input {
        let o1 = l1(self)
        let o2 = l2(o1)
        return l3(o2)
    }

    /// Returns the output computed by applying a sequence of layers to the previous layer's output,
    /// except that the first layer's input is `self`.
    ///
    /// - Parameters:
    ///   - l1: The first layer.
    ///   - l2: The second layer.
    ///   - l3: The third layer.
    ///   - l4: The fourth layer.
    /// - Returns: The final layer's output after sequential application.
    @differentiable
    func sequenced<L1: Layer, L2: Layer, L3: Layer, L4: Layer>(
        through l1: L1, _ l2: L2, _ l3: L3, _ l4: L4
    ) -> L4.Output
        where L1.Input == Self, L1.Output == L2.Input, L2.Output == L3.Input,
              L3.Output == L4.Input {
        let o1 = l1(self)
        let o2 = l2(o1)
        let o3 = l3(o2)
        return l4(o3)
    }

    /// Returns the output computed by applying a sequence of layers to the previous layer's output,
    /// except that the first layer's input is `self`.
    ///
    /// - Parameters:
    ///   - l1: The first layer.
    ///   - l2: The second layer.
    ///   - l3: The third layer.
    ///   - l4: The third layer.
    ///   - l5: The fifth layer.
    /// - Returns: The final layer's output after sequential application.
    @differentiable
    func sequenced<L1: Layer, L2: Layer, L3: Layer, L4: Layer, L5: Layer>(
        through l1: L1, _ l2: L2, _ l3: L3, _ l4: L4, _ l5: L5
    ) -> L5.Output
        where L1.Input == Self, L1.Output == L2.Input, L2.Output == L3.Input, L3.Output == L4.Input,
              L4.Output == L5.Input {
        let o1 = l1(self)
        let o2 = l2(o1)
        let o3 = l3(o2)
        let o4 = l4(o3)
        return l5(o4)
    }

    /// Returns the output computed by applying a sequence of layers to the previous layer's output,
    /// except that the first layer's input is `self`.
    ///
    /// - Parameters:
    ///   - l1: The first layer.
    ///   - l2: The second layer.
    ///   - l3: The third layer.
    ///   - l4: The third layer.
    ///   - l5: The fifth layer.
    ///   - l6: The sixth layer.
    /// - Returns: The final layer's output after sequential application.
    @differentiable
    func sequenced<L1: Layer, L2: Layer, L3: Layer, L4: Layer, L5: Layer, L6: Layer>(
        through l1: L1, _ l2: L2, _ l3: L3, _ l4: L4, _ l5: L5, _ l6: L6
    ) -> L6.Output
        where L1.Input == Self, L1.Output == L2.Input, L2.Output == L3.Input, L3.Output == L4.Input,
              L4.Output == L5.Input, L5.Output == L6.Input {
        let o1 = l1(self)
        let o2 = l2(o1)
        let o3 = l3(o2)
        let o4 = l4(o3)
        let o5 = l5(o4)
        return l6(o5)
    }
}


/// A mutable, shareable, owning reference to a tensor.
public final class Parameter<Scalar: TensorFlowScalar> {
    public var value: Tensor<Scalar>
    public init(_ value: Tensor<Scalar>) {
        self.value = value
    }
}
