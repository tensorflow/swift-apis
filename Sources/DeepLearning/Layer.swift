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
    @differentiable(wrt: (self, input))
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

    // FIXME(SR-9716): Remove this once the bug is fixed or worked around.
    public var allKeyPaths: [PartialKeyPath<Dense>] {
        return [\Dense.weight, \Dense.bias]
    }

    @differentiable(wrt: (self, input))
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        return activation(matmul(input, weight) + bias)
    }
}

public extension Dense where Scalar.RawSignificand: FixedWidthInteger {
    init<G: RandomNumberGenerator>(
        inputSize: Int,
        outputSize: Int,
        activation: @escaping Activation,
        generator: inout G
    ) {
        self.init(weight: Tensor(glorotUniform: [Int32(inputSize), Int32(outputSize)],
                                 generator: &generator),
                  bias: Tensor(zeros: [Int32(outputSize)]),
                  activation: activation)
    }

    init(inputSize: Int, outputSize: Int, activation: @escaping Activation) {
      self.init(inputSize: inputSize, outputSize: outputSize, activation: activation,
                generator: &PhiloxRandomNumberGenerator.global)
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

    @differentiable(wrt: (self, input))
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
            strides: (Int32(strides.0), Int32(strides.1)),
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

    @differentiable(wrt: (self, input))
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

    @differentiable(wrt: (self, input))
    private func applyingInference(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        let inv = rsqrt(runningVariance.value + epsilon) * scale
        return (input - runningMean.value) * inv + offset
    }

    @differentiable(wrt: (self, input), vjp: _vjpApplied(to:in:))
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

    public init(poolSize: (Int, Int), strides: (Int, Int), padding: Padding = .valid) {
        self.poolSize = (1, Int32(poolSize.0), Int32(poolSize.1), 1)
        self.strides = (1, Int32(strides.0), Int32(strides.1), 1)
        self.padding = padding
    }

    @differentiable(wrt: (self, input))
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

    public init(poolSize: (Int, Int), strides: (Int, Int), padding: Padding = .valid) {
        self.poolSize = (1, Int32(poolSize.0), Int32(poolSize.1), 1)
        self.strides = (1, Int32(strides.0), Int32(strides.1), 1)
        self.padding = padding
    }

    @differentiable(wrt: (self, input))
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

    public init(featureCount: Int,
                axis: Int,
                epsilon: Tensor<Scalar> = Tensor(0.001)) {
        self.scale = Tensor<Scalar>(ones: [Int32(featureCount)])
        self.offset = Tensor<Scalar>(zeros: [Int32(featureCount)])
        self.axis = Int32(axis)
        self.epsilon = epsilon
    }

    @differentiable(wrt: (self, input))
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

    @differentiable(wrt: (self, input))
    private func applyingTraining(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.droppingOut(probability: probability)
    }

    @differentiable(wrt: (self, input))
    private func applyingInference(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input
    }

    @differentiable(wrt: (self, input), vjp: _vjpApplied(to:in:))
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

    @differentiable(wrt: (self, input))
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        let shape = input.shape
        let (batchSize, height, width, channels) = (shape[0], shape[1], shape[2], shape[3])
        let reshapeSize = Tensor<Int32>([batchSize, height, 1, width, 1, channels])
        let scaleOnes = Tensor<Scalar>(ones: [1, 1, size, 1, size, 1])
        let upSampling = input.reshaped(toShape: reshapeSize) * scaleOnes
        let upSampledShape = Tensor<Int32>([batchSize, height * size, width * size, channels])
        return upSampling.reshaped(toShape: upSampledShape)
    }
}
