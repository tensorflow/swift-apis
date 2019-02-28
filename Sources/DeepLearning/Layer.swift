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
@_exported import TensorFlow
#endif

/// A value that indicates either a training phase or an inference phase for a layer.
public enum LearningPhase {
    case training
    case inference
}

/// A context that stores contextual information used for the application of layers.
open class Context {
    /// The current learning phase.
    public var learningPhase: LearningPhase

    /// Creates a context.
    ///
    /// - Parameter learningPhase: The current learning phase.
    public required init(learningPhase: LearningPhase) {
        self.learningPhase = learningPhase
    }

    /// Creates a context by copying all information from an existing context.
    ///
    /// - Parameter context: The existing context to copy from.
    public required init(_ other: Context) {
        self.learningPhase = other.learningPhase
    }
}

/// A neural network layer.
///
/// Types that conform to `Layer` represent functions that map inputs to outputs. They may have an
/// internal state represented by parameters, such as weight tensors.
///
/// `Layer` instances define a differentiable `applied(to:in:)` method for mapping inputs to
/// outputs.
public protocol Layer: Differentiable & KeyPathIterable
    where AllDifferentiableVariables: KeyPathIterable {
    /// The input type of the layer.
    associatedtype Input: Differentiable
    /// The output type of the layer.
    associatedtype Output: Differentiable

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters
    ///   - input: The input to the layer.
    ///   - context: The contextual informance for the layer application, e.g. the current learning
    ///     phase.
    /// - Returns: The output.
    @differentiable
    func applied(to input: Input, in context: Context) -> Output
}

public extension Layer {
    @available(*, deprecated,
               message: "Switch to 'applied(to:in:)' for training, or 'inferring(from:)' for inference")
    func applied(to input: Input) -> Output {
        return inferring(from: input)
    }

    /// Returns the inference output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The inference output.
    @differentiable
    func inferring(from input: Input) -> Output {
        let context = Context(learningPhase: .inference)
        return applied(to: input, in: context)
    }

    /// Returns the inference output and the backpropagation function obtained from applying the
    /// layer to the given input. 
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: A tuple containing the output and the backpropagation function. The
    ///   backpropagation function (a.k.a. backpropagator) takes a direction vector and returns the
    ///   gradients at the layer and at the input, respectively.
    func appliedForBackpropagation(to input: Input, in context: Context)
        -> (output: Output,
            backpropagator: (_ direction: Output.CotangentVector)
                -> (layerGradient: CotangentVector, inputGradient: Input.CotangentVector)) {
        let (out, pullback) = valueWithPullback(at: input) { layer, input in
            return layer.applied(to: input, in: context)
        }
        return (out, pullback)
    }
}

/// Adds helpers for standard feed-forward, sequential models.
public extension Differentiable {
    @differentiable
    func sequenced<L1: Layer, L2: Layer>(
        in context: Context, through l1: L1, _ l2: L2)
        -> L2.Output
            where L1.Input == Self,
                  L1.Output == L2.Input {
        let o1 = l1.applied(to: self, in: context)
        return l2.applied(to: o1, in: context)
    }

    @differentiable
    func sequenced<L1: Layer, L2: Layer, L3: Layer>(
        in context: Context, through l1: L1, _ l2: L2, _ l3: L3)
        -> L3.Output
            where L1.Input == Self,
                  L1.Output == L2.Input,
                  L2.Output == L3.Input {
        let o1 = l1.applied(to: self, in: context)
        let o2 = l2.applied(to: o1, in: context)
        return l3.applied(to: o2, in: context)
    }

    @differentiable
    func sequenced<L1: Layer, L2: Layer, L3: Layer, L4: Layer>(
        in context: Context, through l1: L1, _ l2: L2, _ l3: L3, _ l4: L4)
        -> L4.Output
            where L1.Input == Self,
                  L1.Output == L2.Input,
                  L2.Output == L3.Input,
                  L3.Output == L4.Input {
        let o1 = l1.applied(to: self, in: context)
        let o2 = l2.applied(to: o1, in: context)
        let o3 = l3.applied(to: o2, in: context)
        return l4.applied(to: o3, in: context)
    }

    @differentiable
    func sequenced<L1: Layer, L2: Layer, L3: Layer, L4: Layer, L5: Layer>(
        in context: Context, through l1: L1, _ l2: L2, _ l3: L3, _ l4: L4, _ l5: L5)
        -> L5.Output
            where L1.Input == Self,
                  L1.Output == L2.Input,
                  L2.Output == L3.Input,
                  L3.Output == L4.Input,
                  L4.Output == L5.Input {
        let o1 = l1.applied(to: self, in: context)
        let o2 = l2.applied(to: o1, in: context)
        let o3 = l3.applied(to: o2, in: context)
        let o4 = l4.applied(to: o3, in: context)
        return l5.applied(to: o4, in: context)
    }

    @differentiable
    func sequenced<L1: Layer, L2: Layer, L3: Layer, L4: Layer, L5: Layer, L6: Layer>(
        in context: Context, through l1: L1, _ l2: L2, _ l3: L3, _ l4: L4, _ l5: L5, _ l6: L6)
        -> L6.Output
            where L1.Input == Self,
                  L1.Output == L2.Input,
                  L2.Output == L3.Input,
                  L3.Output == L4.Input,
                  L4.Output == L5.Input,
                  L5.Output == L6.Input {
        let o1 = l1.applied(to: self, in: context)
        let o2 = l2.applied(to: o1, in: context)
        let o3 = l3.applied(to: o2, in: context)
        let o4 = l4.applied(to: o3, in: context)
        let o5 = l5.applied(to: o4, in: context)
        return l6.applied(to: o5, in: context)
    }
}


/// A mutable, shareable, owning reference to a tensor.
public final class Parameter<Scalar: TensorFlowScalar> {
    public var value: Tensor<Scalar>
    public init(_ value: Tensor<Scalar>) {
        self.value = value
    }
}

/// A densely-connected neural network layer.
///
/// `Dense` implements the operation `activation(matmul(input, weight) + bias)` where `activation`
/// is the element-wise activation function passed as the activation argument. `weight` is a weight
/// matrix created by the layer, and `bias` is a bias vector created by the layer.
@_fixed_layout
public struct Dense<Scalar: TensorFlowFloatingPoint>: Layer {
    public var weight: Tensor<Scalar>
    public var bias: Tensor<Scalar>
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
    @noDerivative public let activation: Activation

    public init(
        weight: Tensor<Scalar>,
        bias: Tensor<Scalar>,
        activation: @escaping Activation
    ) {
        self.weight = weight
        self.bias = bias
        self.activation = activation
    }

    @differentiable
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        return activation(matmul(input, weight) + bias)
    }
}

public extension Dense where Scalar.RawSignificand: FixedWidthInteger {
    init<G: RandomNumberGenerator>(
        inputSize: Int,
        outputSize: Int,
        activation: @escaping Activation = identity,
        generator: inout G
    ) {
        self.init(weight: Tensor(glorotUniform: [Int32(inputSize), Int32(outputSize)],
                                 generator: &generator),
                  bias: Tensor(zeros: [Int32(outputSize)]),
                  activation: activation)
    }

    init(inputSize: Int, outputSize: Int, activation: @escaping Activation = identity) {
      self.init(inputSize: inputSize, outputSize: outputSize, activation: activation,
                generator: &PhiloxRandomNumberGenerator.global)
    }
}

public extension Dense {
    init(
        inputSize: Int,
        outputSize: Int,
        activation: @escaping Activation = identity,
        seed: (Int64, Int64) = (Int64.random(in: Int64.min..<Int64.max),
                                Int64.random(in: Int64.min..<Int64.max))
    ) {
        self.init(weight: Tensor(glorotUniform: [Int32(inputSize), Int32(outputSize)],
                                 seed: seed),
                  bias: Tensor(zeros: [Int32(outputSize)]),
                  activation: activation)
    }
}

@_fixed_layout
public struct Conv2D<Scalar: TensorFlowFloatingPoint>: Layer {
    public var filter: Tensor<Scalar>
    public var bias: Tensor<Scalar>
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
    @noDerivative public let activation: Activation
    @noDerivative public let strides: (Int32, Int32)
    @noDerivative public let padding: Padding

    public init(
        filter: Tensor<Scalar>,
        bias: Tensor<Scalar>,
        activation: @escaping Activation,
        strides: (Int, Int),
        padding: Padding
    ) {
        self.filter = filter
        self.bias = bias
        self.activation = activation
        (self.strides.0, self.strides.1) = (Int32(strides.0), Int32(strides.1))
        self.padding = padding
    }

    @differentiable
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        return activation(input.convolved2D(withFilter: filter,
                                            strides: (1, strides.0, strides.1, 1),
                                            padding: padding) + bias)
    }
}

public extension Conv2D where Scalar.RawSignificand: FixedWidthInteger {
    init<G: RandomNumberGenerator>(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        activation: @escaping Activation = identity,
        generator: inout G
    ) {
        let filterTensorShape = TensorShape([
            Int32(filterShape.0), Int32(filterShape.1),
            Int32(filterShape.2), Int32(filterShape.3)])
        self.init(
            filter: Tensor(glorotUniform: filterTensorShape),
            bias: Tensor(zeros: TensorShape([Int32(filterShape.3)])),
            activation: activation,
            strides: strides,
            padding: padding)
    }

    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        activation: @escaping Activation = identity
    ) {
      self.init(filterShape: filterShape, strides: strides, padding: padding,
                activation: activation,
                generator: &PhiloxRandomNumberGenerator.global)
    }
}

public extension Conv2D {
    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        activation: @escaping Activation = identity,
        seed: (Int64, Int64) = (Int64.random(in: Int64.min..<Int64.max),
                                Int64.random(in: Int64.min..<Int64.max))
    ) {
        let filterTensorShape = TensorShape([
            Int32(filterShape.0), Int32(filterShape.1),
            Int32(filterShape.2), Int32(filterShape.3)])
        self.init(
          filter: Tensor(glorotUniform: filterTensorShape, seed: seed),
          bias: Tensor(zeros: TensorShape([Int32(filterShape.3)])),
          activation: activation,
          strides: (Int32(strides.0), Int32(strides.1)),
          padding: padding)
    }
}

@_fixed_layout
public struct BatchNorm<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The batch dimension.
    @noDerivative public let axis: Int32
    /// The momentum for the running mean and running variance.
    @noDerivative public let momentum: Tensor<Scalar>
    /// The offset value, also known as beta.
    public var offset: Tensor<Scalar>
    /// The scale value, also known as gamma.
    public var scale: Tensor<Scalar>
    /// The variance epsilon value.
    @noDerivative public let epsilon: Tensor<Scalar>
    /// The running mean.
    @noDerivative public let runningMean: Parameter<Scalar>
    /// The running variance.
    @noDerivative public let runningVariance: Parameter<Scalar>

    /// The batch dimension.
    public init(
        axis: Int,
        momentum: Tensor<Scalar>,
        offset: Tensor<Scalar>,
        scale: Tensor<Scalar>,
        epsilon: Tensor<Scalar>,
        runningMean: Tensor<Scalar>,
        runningVariance: Tensor<Scalar>
    ) {
        self.axis = Int32(axis)
        self.momentum = momentum
        self.offset = offset
        self.scale = scale
        self.epsilon = epsilon
        self.runningMean = Parameter(runningMean)
        self.runningVariance = Parameter(runningVariance)
    }

    @differentiable
    private func applyingTraining(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        let positiveAxis = (input.rank + axis) % input.rank
        let mean = input.mean(alongAxes: [0, positiveAxis])
        let variance = input.variance(alongAxes: [0, positiveAxis])
        runningMean.value += (mean - runningMean.value) * (1 - momentum)
        runningVariance.value += (
            variance - runningVariance.value) * (1 - momentum)
        let inv = rsqrt(variance + epsilon) * scale
        return (input - mean) * inv + offset
    }

    @differentiable
    private func applyingInference(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        let inv = rsqrt(runningVariance.value + epsilon) * scale
        return (input - runningMean.value) * inv + offset
    }

    @differentiable(vjp: _vjpApplied(to:in:))
    public func applied(to input: Tensor<Scalar>, in context: Context) -> Tensor<Scalar> {
        switch context.learningPhase {
        case .training:
            return applyingTraining(to: input)
        case .inference:
            return applyingInference(to: input)
        }
    }

    @usableFromInline
    func _vjpApplied(to input: Tensor<Scalar>, in context: Context) ->
        (Tensor<Scalar>, (Tensor<Scalar>) ->
            (BatchNorm<Scalar>.CotangentVector, Tensor<Scalar>)) {
        switch context.learningPhase {
        case .training:
            return valueWithPullback(at: input) {
                $0.applyingTraining(to: $1)
            }
        case .inference:
            return valueWithPullback(at: input) {
                $0.applyingInference(to: $1)
            }
        }
    }

    public init(featureCount: Int,
                axis: Int = -1,
                momentum: Tensor<Scalar> = Tensor(0.99),
                epsilon: Tensor<Scalar> = Tensor(0.001)) {
        self.axis = Int32(axis)
        self.momentum = momentum
        self.scale = Tensor<Scalar>(ones: [Int32(featureCount)])
        self.offset = Tensor<Scalar>(zeros: [Int32(featureCount)])
        self.epsilon = epsilon
        self.runningMean = Parameter(Tensor(0))
        self.runningVariance = Parameter(Tensor(1))
    }
}

@_fixed_layout
public struct MaxPool2D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The size of the sliding reduction window for pooling.
    @noDerivative let poolSize: (Int32, Int32, Int32, Int32)
    /// The strides of the sliding window for each dimension of a 4-D input.
    /// Strides in non-spatial dimensions must be `1`.
    @noDerivative let strides: (Int32, Int32, Int32, Int32)
    /// The padding algorithm for pooling.
    @noDerivative let padding: Padding

    public init(
        poolSize: (Int, Int, Int, Int),
        strides: (Int, Int, Int, Int),
        padding: Padding
    ) {
        (self.poolSize.0, self.poolSize.1, self.poolSize.2, self.poolSize.3)
            = (Int32(poolSize.0), Int32(poolSize.1), Int32(poolSize.2), Int32(poolSize.3))
        (self.strides.0, self.strides.1, self.strides.2, self.strides.3)
            = (Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3))
        self.padding = padding
    }

    public init(poolSize: (Int, Int), strides: (Int, Int), padding: Padding = .valid) {
        self.poolSize = (1, Int32(poolSize.0), Int32(poolSize.1), 1)
        self.strides = (1, Int32(strides.0), Int32(strides.1), 1)
        self.padding = padding
    }

    @differentiable
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        return input.maxPooled(
          kernelSize: poolSize, strides: strides, padding: padding)
    }
}

@_fixed_layout
public struct AvgPool2D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The size of the sliding reduction window for pooling.
    @noDerivative let poolSize: (Int32, Int32, Int32, Int32)
    /// The strides of the sliding window for each dimension of a 4-D input.
    /// Strides in non-spatial dimensions must be `1`.
    @noDerivative let strides: (Int32, Int32, Int32, Int32)
    /// The padding algorithm for pooling.
    @noDerivative let padding: Padding

    public init(
        poolSize: (Int, Int, Int, Int),
        strides: (Int, Int, Int, Int),
        padding: Padding
    ) {
        (self.poolSize.0, self.poolSize.1, self.poolSize.2, self.poolSize.3)
            = (Int32(poolSize.0), Int32(poolSize.1), Int32(poolSize.2), Int32(poolSize.3))
        (self.strides.0, self.strides.1, self.strides.2, self.strides.3)
            = (Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3))
        self.padding = padding
    }

    public init(poolSize: (Int, Int), strides: (Int, Int), padding: Padding = .valid) {
        self.poolSize = (1, Int32(poolSize.0), Int32(poolSize.1), 1)
        self.strides = (1, Int32(strides.0), Int32(strides.1), 1)
        self.padding = padding
    }

    @differentiable
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        return input.averagePooled(
          kernelSize: poolSize, strides: strides, padding: padding)
    }
}

@_fixed_layout
public struct LayerNorm<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The offset value, also known as beta.
    public var offset: Tensor<Scalar>
    /// The scale value, also known as gamma.
    public var scale: Tensor<Scalar>
    /// The axis.
    @noDerivative public let axis: Int32
    /// The variance epsilon value.
    @noDerivative public let epsilon: Tensor<Scalar>

    public init(
        offset: Tensor<Scalar>,
        scale: Tensor<Scalar>,
        axis: Int,
        epsilon: Tensor<Scalar>
    ) {
        self.offset = offset
        self.scale = scale
        self.axis = Int32(axis)
        self.epsilon = epsilon
    }

    public init(featureCount: Int,
                axis: Int,
                epsilon: Tensor<Scalar> = Tensor(0.001)) {
        self.init(
            offset: Tensor(zeros: [Int32(featureCount)]),
            scale: Tensor(ones: [Int32(featureCount)]),
            axis: axis,
            epsilon: epsilon
        )
    }

    @differentiable
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        let mean = input.mean(alongAxes: axis)
        let variance = input.variance(alongAxes: axis)
        let inv = rsqrt(variance + epsilon) * scale
        return (input - mean) * inv + offset
    }
}

public extension Tensor
    where Scalar: TensorFlowFloatingPoint, Scalar.RawSignificand: FixedWidthInteger {
    @differentiable(wrt: self where Scalar: Differentiable)
    func droppingOut(probability: Double) -> Tensor {
        let noise = Tensor(randomUniform: shape)
        let keepMask = noise .>= Scalar(probability)
        let keepProbability = Scalar(1.0 - probability)
        return self * Tensor(keepMask) / Tensor(keepProbability)
    }
}

@_fixed_layout
public struct Dropout<Scalar: TensorFlowFloatingPoint>: Layer
    where Scalar.RawSignificand: FixedWidthInteger {
    @noDerivative public let probability: Double

    public init(probability: Double) {
        self.probability = probability
    }

    @differentiable
    private func applyingTraining(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.droppingOut(probability: probability)
    }

    @differentiable
    private func applyingInference(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input
    }

    @differentiable(vjp: _vjpApplied(to:in:))
    public func applied(to input: Tensor<Scalar>, in context: Context) -> Tensor<Scalar> {
        switch context.learningPhase {
        case .training:
            return applyingTraining(to: input)
        case .inference:
            return applyingInference(to: input)
        }
    }

    @usableFromInline
    func _vjpApplied(to input: Tensor<Scalar>, in context: Context) ->
        (Tensor<Scalar>, (Tensor<Scalar>) ->
            (Dropout<Scalar>.CotangentVector, Tensor<Scalar>)) {
        switch context.learningPhase {
        case .training:
            return valueWithPullback(at: input) {
                $0.applyingTraining(to: $1)
            }
        case .inference:
            return valueWithPullback(at: input) {
                $0.applyingInference(to: $1)
            }
        }
    }
}

@_fixed_layout
public struct UpSampling2D<Scalar: TensorFlowFloatingPoint>: Layer {
    @noDerivative public let size: Int32

    public init(size: Int32) {
       self.size = size
    }

    @differentiable
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        let shape = input.shape
        let (batchSize, height, width, channels) = (shape[0], shape[1], shape[2], shape[3])
        let scaleOnes = Tensor<Scalar>(ones: [1, 1, size, 1, size, 1])
        let upSampling = input.reshaped(to: [batchSize, height, 1, width, 1, channels]) * scaleOnes
        return upSampling.reshaped(to: [batchSize, height * size, width * size, channels])
    }
}

@_fixed_layout
public struct Flatten<Scalar: TensorFlowFloatingPoint>: Layer {
    @differentiable
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        let batchSize = input.shape[0]
        let remaining = input.shape[1..<input.rank].contiguousSize
        return input.reshaped(to: [batchSize, remaining])
    }
}

@_fixed_layout
public struct Reshape<Scalar: TensorFlowFloatingPoint>: Layer {
    @noDerivative public let shape: Tensor<Int32>
    
    public init(shape: Tensor<Int32>) {
        self.shape = shape
    }
    
    public init(_ shape: TensorShape) {
        self.init(shape: Tensor(shape.dimensions))
    }
    
    @differentiable
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        return input.reshaped(toShape: shape)
    }
}
