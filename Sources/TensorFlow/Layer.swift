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

import _Differentiation

public protocol Module: EuclideanDifferentiable, KeyPathIterable
where
  TangentVector: VectorProtocol & ElementaryFunctions & PointwiseMultiplicative & KeyPathIterable
{
  /// The input type of the layer.
  associatedtype Input
  /// The output type of the layer.
  associatedtype Output: Differentiable

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable(wrt: self)
  func callAsFunction(_ input: Input) -> Output

  @differentiable(wrt: self)
  func forward(_ input: Input) -> Output
}

extension Module {
  @differentiable(wrt: self)
  public func forward(_ input: Input) -> Output {
    return callAsFunction(input)
  }
}

extension Module where Input: TensorProtocol, Output: DifferentiableTensorProtocol {
  @differentiable(wrt: self)
  public func callAsFunction(_ input: Input) -> Output {
    let activation = forward(input)

    return annotated(activation)
  }

  @differentiable
  public func annotated(_ output: Output) -> Output {
    #if USING_X10_BACKEND
      let selfType = String(describing: Self.self)
      let annotation = "type=" + selfType
      let annotated = output.annotate(annotation)
      return annotated
    #else
      return output
    #endif
  }

  /// Returns the annotations obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: All collected annotations from the XLA graph.
  public func annotations(input: Input) -> String {
    let output = self.callAsFunction(input)
    return output.annotations
  }

  public func summary(input: Input) -> String {
    return self.annotations(input: input)
  }
}

/// A neural network layer.
///
/// Types that conform to `Layer` represent functions that map inputs to outputs. They may have an
/// internal state represented by parameters, such as weight tensors.
///
/// `Layer` instances define a differentiable `callAsFunction(_:)` method for mapping inputs to
/// outputs.
public protocol Layer: Module where Input: Differentiable {
  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable
  func callAsFunction(_ input: Input) -> Output

  @differentiable
  func forward(_ input: Input) -> Output
}

extension Layer {
  // Workaround for SR-13455: autodiff undefined symbol linker error.
  @differentiable(wrt: self)
  @differentiable
  public func forward(_ input: Input) -> Output {
    return callAsFunction(input)
  }
}

extension Layer where Input: DifferentiableTensorProtocol, Output: DifferentiableTensorProtocol {
  // Workaround for SR-13455: autodiff undefined symbol linker error.
  @differentiable(wrt: self)
  @differentiable
  public func callAsFunction(_ input: Input) -> Output {
    let activation = forward(input)
    return annotated(activation)
  }

  /// Returns the annotations obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: All collected annotations from the XLA graph.
  public func annotations(input: Input) -> String {
    return self.annotations(inputShape: input.shape)
  }

  /// Returns the annotations obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The shape of the input to the layer.
  /// - Returns: All collected annotations from the XLA graph.
  public func annotations(inputShape: TensorShape) -> String {
    #if USING_X10_BACKEND
      LazyTensorBarrier()
      let zeros = Input(repeating: 0, shape: inputShape, on: Device.defaultXLA)
      let model = Self(copying: self, to: Device.defaultXLA)
      let output = model(zeros)

      return output.annotations
    #else
      return ""
    #endif
  }

  public func summary(inputShape: TensorShape) -> String {
    return self.annotations(inputShape: inputShape)
  }
}

/// An empty struct representing empty `TangentVector`s for parameterless layers.
public struct EmptyTangentVector: EuclideanDifferentiable, VectorProtocol, ElementaryFunctions,
  PointwiseMultiplicative, KeyPathIterable
{
  public typealias VectorSpaceScalar = Float
  public typealias TangentVector = Self

  public init() {}

  public func adding(_ x: Float) -> EmptyTangentVector { self }
  public mutating func add(_ x: Float) {}
  public func subtracting(_ x: Float) -> EmptyTangentVector { self }
  public mutating func subtract(_ x: Float) {}
  public func scaled(by scalar: Float) -> EmptyTangentVector { self }
  public mutating func scale(by scalar: Float) {}
}

/// A parameterless neural network layer.
///
/// The `TangentVector` of parameterless layers is always `EmptyTangentVector`.
public protocol ParameterlessLayer: Layer where TangentVector == EmptyTangentVector {
  @differentiable
  func callAsFunction(_ input: Input) -> Output
}

extension ParameterlessLayer {
  public mutating func move(along direction: EmptyTangentVector) {}
  public var differentiableVectorView: EmptyTangentVector { EmptyTangentVector() }
}

extension Layer {
  /// Returns the inference output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The inference output.
  public func inferring(from input: Input) -> Output {
    withLearningPhase(LearningPhase.inference) { self(input) }
  }

  // TODO(TF-433, SR-11882): Remove this custom derivative when
  // differentiation supports `rethrows` functions and currying.
  @usableFromInline
  @derivative(of: inferring(from:))
  internal func _vjpInferring(from input: Input)
    -> (
      value: Output,
      pullback: (Output.TangentVector)
        -> (TangentVector, Input.TangentVector)
    )
  {
    withLearningPhase(LearningPhase.inference) {
      let (output, pullback) = appliedForBackpropagation(to: input)
      return (output, { v in pullback(v) })
    }
  }

  public typealias Backpropagator = (_ direction: Output.TangentVector)
    -> (layerGradient: TangentVector, inputGradient: Input.TangentVector)

  /// Returns the inference output and the backpropagation function obtained from applying the
  /// layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: A tuple containing the output and the backpropagation function. The
  ///   backpropagation function (a.k.a. backpropagator) takes a direction vector and returns the
  ///   gradients at the layer and at the input, respectively.
  public func appliedForBackpropagation(to input: Input)
    -> (output: Output, backpropagator: Backpropagator)
  {
    let (out, pullback) = Swift.valueWithPullback(at: self, input) { layer, input in
      return layer(input)
    }
    return (out, pullback)
  }
}

extension Differentiable {
  /// Returns the output computed by applying a sequence of layers to the previous layer's output,
  /// except that the first layer's input is `self`.
  ///
  /// - Parameters:
  ///   - l1: The first layer.
  ///   - l2: The second layer.
  /// - Returns: The final layer's output after sequential application.
  @differentiable
  public func sequenced<L1: Layer, L2: Layer>(through l1: L1, _ l2: L2) -> L2.Output
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
  public func sequenced<L1: Layer, L2: Layer, L3: Layer>(through l1: L1, _ l2: L2, _ l3: L3)
    -> L3.Output
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
  public func sequenced<L1: Layer, L2: Layer, L3: Layer, L4: Layer>(
    through l1: L1, _ l2: L2, _ l3: L3, _ l4: L4
  ) -> L4.Output
  where
    L1.Input == Self, L1.Output == L2.Input, L2.Output == L3.Input,
    L3.Output == L4.Input
  {
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
  public func sequenced<L1: Layer, L2: Layer, L3: Layer, L4: Layer, L5: Layer>(
    through l1: L1, _ l2: L2, _ l3: L3, _ l4: L4, _ l5: L5
  ) -> L5.Output
  where
    L1.Input == Self, L1.Output == L2.Input, L2.Output == L3.Input, L3.Output == L4.Input,
    L4.Output == L5.Input
  {
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
  public func sequenced<L1: Layer, L2: Layer, L3: Layer, L4: Layer, L5: Layer, L6: Layer>(
    through l1: L1, _ l2: L2, _ l3: L3, _ l4: L4, _ l5: L5, _ l6: L6
  ) -> L6.Output
  where
    L1.Input == Self, L1.Output == L2.Input, L2.Output == L3.Input, L3.Output == L4.Input,
    L4.Output == L5.Input, L5.Output == L6.Input
  {
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
