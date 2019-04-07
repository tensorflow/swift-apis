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
    /// - Parameters:
    ///   - input: The input to the layer.
    ///   - context: The contextual information for the layer application, e.g. the current learning
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

    typealias Backpropagator = (_ direction: Output.CotangentVector)
        -> (layerGradient: CotangentVector, inputGradient: Input.CotangentVector)

    /// Returns the inference output and the backpropagation function obtained from applying the
    /// layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: A tuple containing the output and the backpropagation function. The
    ///   backpropagation function (a.k.a. backpropagator) takes a direction vector and returns the
    ///   gradients at the layer and at the input, respectively.
    func appliedForBackpropagation(to input: Input, in context: Context)
        -> (output: Output, backpropagator: Backpropagator) {
        let (out, pullback) = valueWithPullback(at: input) { layer, input in
            return layer.applied(to: input, in: context)
        }
        return (out, pullback)
    }
}

public extension Differentiable {
    /// Returns the output computed by applying a sequence of layers to the previous layer's output,
    /// except that the first layer's input is `self`.
    ///
    /// - Parameters:
    ///   - context: The context that stores contextual information used for the application of
    ///     layers.
    ///   - l1: The first layer.
    ///   - l2: The second layer.
    /// - Returns: The final layer's output after sequential application.
    @differentiable
    func sequenced<L1: Layer, L2: Layer>(
        in context: Context, through l1: L1, _ l2: L2)
        -> L2.Output
            where L1.Input == Self,
                  L1.Output == L2.Input {
        let o1 = l1.applied(to: self, in: context)
        return l2.applied(to: o1, in: context)
    }

    /// Returns the output computed by applying a sequence of layers to the previous layer's output,
    /// except that the first layer's input is `self`.
    ///
    /// - Parameters:
    ///   - context: The context that stores contextual information used for the application of
    ///     layers.
    ///   - l1: The first layer.
    ///   - l2: The second layer.
    ///   - l3: The third layer.
    /// - Returns: The final layer's output after sequential application.
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

    /// Returns the output computed by applying a sequence of layers to the previous layer's output,
    /// except that the first layer's input is `self`.
    ///
    /// - Parameters:
    ///   - context: The context that stores contextual information used for the application of
    ///     layers.
    ///   - l1: The first layer.
    ///   - l2: The second layer.
    ///   - l3: The third layer.
    ///   - l4: The fourth layer.
    /// - Returns: The final layer's output after sequential application.
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

    /// Returns the output computed by applying a sequence of layers to the previous layer's output,
    /// except that the first layer's input is `self`.
    ///
    /// - Parameters:
    ///   - context: The context that stores contextual information used for the application of
    ///     layers.
    ///   - l1: The first layer.
    ///   - l2: The second layer.
    ///   - l3: The third layer.
    ///   - l4: The third layer.
    ///   - l5: The fifth layer.
    /// - Returns: The final layer's output after sequential application.
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

    /// Returns the output computed by applying a sequence of layers to the previous layer's output,
    /// except that the first layer's input is `self`.
    ///
    /// - Parameters:
    ///   - context: The context that stores contextual information used for the application of
    ///     layers.
    ///   - l1: The first layer.
    ///   - l2: The second layer.
    ///   - l3: The third layer.
    ///   - l4: The third layer.
    ///   - l5: The fifth layer.
    ///   - l6: The sixth layer.
    /// - Returns: The final layer's output after sequential application.
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
/// `Dense` implements the operation `activation(matmul(input, weight) + bias)`, where `weight` is
/// a weight matrix, `bias` is a bias vector, and `activation` is an element-wise activation
/// function.
@_fixed_layout
public struct Dense<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The weight matrix.
    public var weight: Tensor<Scalar>
    /// The bias vector.
    public var bias: Tensor<Scalar>
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
    /// The element-wise activation function.
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

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    ///   - context: The contextual information for the layer application, e.g. the current learning
    ///     phase.
    /// - Returns: The output.
    @differentiable
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        return activation(matmul(input, weight) + bias)
    }
}

public extension Dense {
    /// Creates a `Dense` layer with the specified input size, output size, and element-wise
    /// activation function. The weight matrix is created with shape `[inputSize, outputSize]` and
    /// is initialized using Glorot uniform initialization with the specified generator. The bias
    /// vector is created with shape `[outputSize]` and is initialized with zeros.
    ///
    /// - Parameters:
    ///   - inputSize: The dimensionality of the input space.
    ///   - outputSize: The dimensionality of the output space.
    ///   - activation: The activation function to use. The default value is `identity(_:)`.
    ///   - generator: The random number generator for initialization.
    ///
    /// - Note: Use `init(inputSize:outputSize:activation:seed:)` for faster random initialization.
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
    /// Creates a `Dense` layer with the specified input size, output size, and element-wise
    /// activation function. The weight matrix is created with shape `[inputSize, outputSize]` and
    /// is initialized using Glorot uniform initialization with the specified seed. The bias vector
    /// is created with shape `[outputSize]` and is initialized with zeros.
    ///
    /// - Parameters:
    ///   - inputSize: The dimensionality of the input space.
    ///   - outputSize: The dimensionality of the output space.
    ///   - activation: The activation function to use. The default value is `identity(_:)`.
    ///   - seed: The random seed for initialization. The default value is random.
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

/// A 1-D convolution layer (e.g. temporal convolution over a time-series).
///
/// This layer creates a convolution filter that is convolved with the layer input to produce a
/// tensor of outputs.
@_fixed_layout
public struct Conv1D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The 3-D convolution kernel `[width, inputChannels, outputChannels]`.
    public var filter: Tensor<Scalar>
    /// The bias vector `[outputChannels]`.
    public var bias: Tensor<Scalar>
    /// An activation function.
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
    /// The element-wise activation function.
    @noDerivative public let activation: Activation
    /// The stride of the sliding window for temporal dimension.
    @noDerivative public let stride: Int32
    /// The padding algorithm for convolution.
    @noDerivative public let padding: Padding

    /// Creates a `Conv1D` layer with the specified filter, bias, activation function, stride, and
    /// padding.
    ///
    /// - Parameters:
    ///   - filter: The 3-D convolution kernel `[width, inputChannels, outputChannels]`.
    ///   - bias: The bias vector `[outputChannels]`.
    ///   - activation: The element-wise activation function.
    ///   - stride: The stride of the sliding window for temporal dimension.
    ///   - padding: The padding algorithm for convolution.
    public init(
        filter: Tensor<Scalar>,
        bias: Tensor<Scalar>,
        activation: @escaping Activation,
        stride: Int,
        padding: Padding
    ) {
        self.filter = filter
        self.bias = bias
        self.activation = activation
        self.stride = Int32(stride)
        self.padding = padding
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer `[batchCount, width, inputChannels]`.
    ///   - context: The contextual information for the layer application, e.g. the current learning
    ///     phase.
    /// - Returns: The output `[batchCount, newWidth, outputChannels]`.
    @differentiable
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        let conv2D = input.expandingShape(at: 1).convolved2D(
            withFilter: filter.expandingShape(at: 0), strides: (1, 1, stride, 1), padding: padding)
        return activation(conv2D.squeezingShape(at: 1) + bias)
    }
}

public extension Conv1D where Scalar.RawSignificand: FixedWidthInteger {
    /// Creates a `Conv1D` layer with the specified filter shape, stride, padding, and
    /// element-wise activation function. The filter tensor is initialized using Glorot uniform
    /// initialization with the specified generator. The bias vector is initialized with zeros.
    ///
    /// - Parameters:
    ///   - filterShape: The 3-D shape of the filter, representing
    ///     `[width, inputChannels, outputChannels]`.
    ///   - stride: The stride of the sliding window for temporal dimension.
    ///   - padding: The padding algorithm for convolution.
    ///   - activation: The element-wise activation function.
    ///   - generator: The random number generator for initialization.
    ///
    /// - Note: Use `init(filterShape:stride:padding:activation:seed:)` for faster random
    ///   initialization.
    init<G: RandomNumberGenerator>(
        filterShape: (Int, Int, Int),
        stride: Int = 1,
        padding: Padding = .valid,
        activation: @escaping Activation = identity,
        generator: inout G
    ) {
        let filterTensorShape = TensorShape([
            Int32(filterShape.0), Int32(filterShape.1), Int32(filterShape.2)])
        self.init(
            filter: Tensor(glorotUniform: filterTensorShape),
            bias: Tensor(zeros: TensorShape([Int32(filterShape.2)])),
            activation: activation,
            stride: stride,
            padding: padding)
    }
}

public extension Conv1D {
    /// Creates a `Conv1D` layer with the specified filter shape, strides, padding, and
    /// element-wise activation function. The filter tensor is initialized using Glorot uniform
    /// initialization with the specified seed. The bias vector is initialized with zeros.
    ///
    /// - Parameters:
    ///   - filterShape: The 3-D shape of the filter, representing
    ///     `[width, inputChannels, outputChannels]`.
    ///   - stride: The stride of the sliding window for temporal dimension.
    ///   - padding: The padding algorithm for convolution.
    ///   - activation: The element-wise activation function.
    ///   - seed: The random seed for initialization. The default value is random.
    init(
        filterShape: (Int, Int, Int),
        stride: Int = 1,
        padding: Padding = .valid,
        activation: @escaping Activation = identity,
        seed: (Int64, Int64) = (Int64.random(in: Int64.min..<Int64.max),
                                Int64.random(in: Int64.min..<Int64.max))
    ) {
        let filterTensorShape = TensorShape([
            Int32(filterShape.0), Int32(filterShape.1), Int32(filterShape.2)])
        self.init(
            filter: Tensor(glorotUniform: filterTensorShape, seed: seed),
            bias: Tensor(zeros: TensorShape([Int32(filterShape.2)])),
            activation: activation,
            stride: Int32(stride),
            padding: padding)
    }
}

/// A 2-D convolution layer (e.g. spatial convolution over images).
///
/// This layer creates a convolution filter that is convolved with the layer input to produce a
/// tensor of outputs.
@_fixed_layout
public struct Conv2D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The 4-D convolution kernel.
    public var filter: Tensor<Scalar>
    /// The bias vector.
    public var bias: Tensor<Scalar>
    /// An activation function.
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
    /// The element-wise activation function.
    @noDerivative public let activation: Activation
    /// The strides of the sliding window for spatial dimensions.
    @noDerivative public let strides: (Int32, Int32)
    /// The padding algorithm for convolution.
    @noDerivative public let padding: Padding

    /// Creates a `Conv2D` layer with the specified filter, bias, activation function, strides, and
    /// padding.
    ///
    /// - Parameters:
    ///   - filter: The 4-D convolution kernel.
    ///   - bias: The bias vector.
    ///   - activation: The element-wise activation function.
    ///   - strides: The strides of the sliding window for spatial dimensions.
    ///   - padding: The padding algorithm for convolution.
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

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    ///   - context: The contextual information for the layer application, e.g. the current learning
    ///     phase.
    /// - Returns: The output.
    @differentiable
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        return activation(input.convolved2D(withFilter: filter,
                                            strides: (1, strides.0, strides.1, 1),
                                            padding: padding) + bias)
    }
}

public extension Conv2D {
    /// Creates a `Conv2D` layer with the specified filter shape, strides, padding, and
    /// element-wise activation function. The filter tensor is initialized using Glorot uniform
    /// initialization with the specified generator. The bias vector is initialized with zeros.
    ///
    /// - Parameters:
    ///   - filterShape: The shape of the 4-D convolution kernel.
    ///   - strides: The strides of the sliding window for spatial dimensions.
    ///   - padding: The padding algorithm for convolution.
    ///   - activation: The element-wise activation function.
    ///   - generator: The random number generator for initialization.
    ///
    /// - Note: Use `init(filterShape:strides:padding:activation:seed:)` for faster random
    ///   initialization.
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
            filter: Tensor(glorotUniform: filterTensorShape, generator: &generator),
            bias: Tensor(zeros: TensorShape([Int32(filterShape.3)])),
            activation: activation,
            strides: strides,
            padding: padding)
    }
}

public extension Conv2D {
    /// Creates a `Conv2D` layer with the specified filter shape, strides, padding, and
    /// element-wise activation function. The filter tensor is initialized using Glorot uniform
    /// initialization with the specified seed. The bias vector is initialized with zeros.
    ///
    /// - Parameters:
    ///   - filterShape: The shape of the 4-D convolution kernel.
    ///   - strides: The strides of the sliding window for spatial dimensions.
    ///   - padding: The padding algorithm for convolution.
    ///   - activation: The element-wise activation function.
    ///   - seed: The random seed for initialization. The default value is random.
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

/// A 2-D transposed convolution layer (e.g. spatial transposed convolution over images).
///
/// This layer creates a convolution filter that is transpose-convolved with the layer input
/// to produce a tensor of outputs.
@_fixed_layout
public struct TransposedConv2D: Layer {
    /// The 4-D convolution kernel.
    public var filter: Tensor<Float>
    /// The bias vector.
    public var bias: Tensor<Float>
    /// An activation function.
    public typealias Activation = @differentiable (Tensor<Float>) -> Tensor<Float>
    /// The element-wise activation function.
    @noDerivative public let activation: Activation
    /// The strides of the sliding window for spatial dimensions.
    @noDerivative public let strides: (Int32, Int32)
    /// The padding algorithm for convolution.
    @noDerivative public let padding: Padding
    @noDerivative public let paddingIndex: Int32

    /// Creates a `TransposedConv2D` layer with the specified filter, bias,
    /// activation function, strides, and padding.
    ///
    /// - Parameters:
    ///   - filter: The 4-D convolution kernel.
    ///   - bias: The bias vector.
    ///   - activation: The element-wise activation function.
    ///   - strides: The strides of the sliding window for spatial dimensions.
    ///   - padding: The padding algorithm for convolution.
    public init(
        filter: Tensor<Float>,
        bias: Tensor<Float>,
        activation: @escaping Activation,
        strides: (Int, Int),
        padding: Padding
    ) {
        self.filter = filter
        self.bias = bias
        self.activation = activation
        (self.strides.0, self.strides.1) = (Int32(strides.0), Int32(strides.1))
        self.padding = padding
        self.paddingIndex = padding == .same ? 0 : 1
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    ///   - context: The contextual information for the layer application, e.g. the current learning
    ///     phase.
    /// - Returns: The output.
    @differentiable
    public func applied(to input: Tensor<Float>, in _: Context) -> Tensor<Float> {
        let batchSize = input.shape[0]
        let w = (input.shape[1] - (1 * paddingIndex)) * strides.0 + (filter.shape[0] * paddingIndex)
        let h = (input.shape[2] - (1 * paddingIndex)) * strides.1 + (filter.shape[1] * paddingIndex)
        let c = filter.shape[2]
        let newShape = Tensor<Int32>([batchSize, w, h, c])
        return activation(input.conv2DBackpropInput(shape: newShape, filter: filter,
                                                         strides: (1, strides.0, strides.1, 1),
                                                         padding: padding) + bias)
    }
}

public extension TransposedConv2D {
    /// Creates a `TransposedConv2D` layer with the specified filter shape, strides, padding, and
    /// element-wise activation function. The filter tensor is initialized using Glorot uniform
    /// initialization with the specified generator. The bias vector is initialized with zeros.
    ///
    /// - Parameters:
    ///   - filterShape: The shape of the 4-D convolution kernel.
    ///   - strides: The strides of the sliding window for spatial dimensions.
    ///   - padding: The padding algorithm for convolution.
    ///   - activation: The element-wise activation function.
    ///   - generator: The random number generator for initialization.
    ///
    /// - Note: Use `init(filterShape:strides:padding:activation:seed:)` for faster random
    ///   initialization.
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
            filter: Tensor(glorotUniform: filterTensorShape, generator: &generator),
            bias: Tensor(zeros: TensorShape([Int32(filterShape.3)])),
            activation: activation,
            strides: strides,
            padding: padding)
    }
}

public extension TransposedConv2D {
    /// Creates a `TransposedConv2D` layer with the specified filter shape, strides, padding, and
    /// element-wise activation function. The filter tensor is initialized using Glorot uniform
    /// initialization with the specified seed. The bias vector is initialized with zeros.
    ///
    /// - Parameters:
    ///   - filterShape: The shape of the 4-D convolution kernel.
    ///   - strides: The strides of the sliding window for spatial dimensions.
    ///   - padding: The padding algorithm for convolution.
    ///   - activation: The element-wise activation function.
    ///   - seed: The random seed for initialization. The default value is random.
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
            strides: strides,
            padding: padding)
    }
}

/// A batch normalization layer.
///
/// Normalizes the activations of the previous layer at each batch, i.e. applies a transformation
/// that maintains the mean activation close to `0` and the activation standard deviation close to
/// `1`.
///
/// Reference: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal
/// Covariate Shift](https://arxiv.org/abs/1502.03167).
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

    /// Creates a batch normalization layer.
    ///
    /// - Parameters:
    ///   - axis: The axis that should be normalized (typically the features axis).
    ///   - momentum: The momentum for the moving average.
    ///   - offset: The offset to be added to the normalized tensor.
    ///   - scale: The scale to multiply the normalized tensor by.
    ///   - epsilon: A small scalar added to the denominator to improve numerical stability.
    ///   - runningMean: The running mean.
    ///   - runningVariance: The running variance.
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

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    ///   - context: The contextual information for the layer application, e.g. the current learning
    ///     phase.
    /// - Returns: The output.
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

    /// Creates a batch normalization layer.
    ///
    /// - Parameters:
    ///   - featureCount: The number of features.
    ///   - axis: The axis that should be normalized (typically the features axis).
    ///   - momentum: The momentum for the moving average.
    ///   - epsilon: A small scalar added to the denominator to improve numerical stability.
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

/// A max pooling layer for temporal data.
@_fixed_layout
public struct MaxPool1D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The size of the sliding reduction window for pooling.
    @noDerivative let poolSize: Int32
    /// The stride of the sliding window for temporal dimension.
    @noDerivative let stride: Int32
    /// The padding algorithm for pooling.
    @noDerivative let padding: Padding

    /// Creates a max pooling layer.
    ///
    /// - Parameters:
    ///   - poolSize: The size of the sliding reduction window for pooling.
    ///   - stride: The stride of the sliding window for temporal dimension.
    ///   - padding: The padding algorithm for pooling.
    public init(
        poolSize: Int,
        stride: Int,
        padding: Padding
    ) {
        self.poolSize = Int32(poolSize)
        self.stride = Int32(stride)
        self.padding = padding
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    ///   - context: The contextual information for the layer application, e.g. the current learning
    ///     phase.
    /// - Returns: The output.
    @differentiable
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        return input.expandingShape(at: 1).maxPooled(
            kernelSize: (1, 1, poolSize, 1), strides: (1, 1, stride, 1), padding: padding
        ).squeezingShape(at: 1)
    }
}

/// A max pooling layer for spatial data.
@_fixed_layout
public struct MaxPool2D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The size of the sliding reduction window for pooling.
    @noDerivative let poolSize: (Int32, Int32, Int32, Int32)
    /// The strides of the sliding window for each dimension of a 4-D input.
    /// Strides in non-spatial dimensions must be `1`.
    @noDerivative let strides: (Int32, Int32, Int32, Int32)
    /// The padding algorithm for pooling.
    @noDerivative let padding: Padding

    /// Creates a max pooling layer.
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

    /// Creates a max pooling layer.
    ///
    /// - Parameters:
    ///   - poolSize: Vertical and horizontal factors by which to downscale.
    ///   - strides: The strides.
    ///   - padding: The padding.
    public init(poolSize: (Int, Int), strides: (Int, Int), padding: Padding = .valid) {
        self.poolSize = (1, Int32(poolSize.0), Int32(poolSize.1), 1)
        self.strides = (1, Int32(strides.0), Int32(strides.1), 1)
        self.padding = padding
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    ///   - context: The contextual information for the layer application, e.g. the current learning
    ///     phase.
    /// - Returns: The output.
    @differentiable
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        return input.maxPooled(
            kernelSize: poolSize, strides: strides, padding: padding)
    }
}

/// An average pooling layer for temporal data.
@_fixed_layout
public struct AvgPool1D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The size of the sliding reduction window for pooling.
    @noDerivative let poolSize: Int32
    /// The stride of the sliding window for temporal dimension.
    @noDerivative let stride: Int32
    /// The padding algorithm for pooling.
    @noDerivative let padding: Padding

    /// Creates an average pooling layer.
    ///
    /// - Parameters:
    ///   - poolSize: The size of the sliding reduction window for pooling.
    ///   - stride: The stride of the sliding window for temporal dimension.
    ///   - padding: The padding algorithm for pooling.
    public init(
        poolSize: Int,
        stride: Int,
        padding: Padding
    ) {
        self.poolSize = Int32(poolSize)
        self.stride = Int32(stride)
        self.padding = padding
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    ///   - context: The contextual information for the layer application, e.g. the current learning
    ///     phase.
    /// - Returns: The output.
    @differentiable
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        return input.expandingShape(at: 1).averagePooled(
            kernelSize: (1, 1, poolSize, 1), strides: (1, 1, stride, 1), padding: padding
        ).squeezingShape(at: 1)
    }
}

/// An average pooling layer for spatial data.
@_fixed_layout
public struct AvgPool2D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The size of the sliding reduction window for pooling.
    @noDerivative let poolSize: (Int32, Int32, Int32, Int32)
    /// The strides of the sliding window for each dimension of a 4-D input.
    /// Strides in non-spatial dimensions must be `1`.
    @noDerivative let strides: (Int32, Int32, Int32, Int32)
    /// The padding algorithm for pooling.
    @noDerivative let padding: Padding

    /// Creates a average pooling layer.
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

    /// Creates a average pooling layer.
    ///
    /// - Parameters:
    ///   - poolSize: Vertical and horizontal factors by which to downscale.
    ///   - strides: The strides.
    ///   - padding: The padding.
    public init(poolSize: (Int, Int), strides: (Int, Int), padding: Padding = .valid) {
        self.poolSize = (1, Int32(poolSize.0), Int32(poolSize.1), 1)
        self.strides = (1, Int32(strides.0), Int32(strides.1), 1)
        self.padding = padding
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    ///   - context: The contextual information for the layer application, e.g. the current learning
    ///     phase.
    /// - Returns: The output.
    @differentiable
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        return input.averagePooled(kernelSize: poolSize, strides: strides, padding: padding)
    }
}


/// A global average pooling layer for temporal data.
@_fixed_layout
public struct GlobalAveragePooling1D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// Creates a global average pooling layer.
    public init() {}

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    ///   - context: The contextual information for the layer application, e.g. the current learning
    ///     phase.
    /// - Returns: The output.
    @differentiable
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        return input.mean(alongAxes: 1).reshaped(to: [input.shape[0], input.shape[2]])
    }
}

/// A global average pooling layer for spatial data.
@_fixed_layout
public struct GlobalAveragePooling2D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// Creates a global average pooling layer.
    public init() {}

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    ///   - context: The contextual information for the layer application, e.g. the current learning
    ///     phase.
    /// - Returns: The output.
    @differentiable
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        return input.mean(alongAxes: [1, 2]).reshaped(to: [input.shape[0], input.shape[3]])
    }
}

/// A global average pooling layer for spatial and spatio-temporal data.
@_fixed_layout
public struct GlobalAveragePooling3D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// Creates a global average pooling layer.
    public init() {}

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    ///   - context: The contextual information for the layer application, e.g. the current learning
    ///     phase.
    /// - Returns: The output.
    @differentiable
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        return input.mean(alongAxes: [1, 2, 3]).reshaped(to: [input.shape[0], input.shape[4]])
    }
}

/// A layer that applies layer normalization over a mini-batch of inputs.
///
/// Reference: [Layer Normalization](https://arxiv.org/abs/1607.06450).
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

    /// Creates a layer normalization layer.
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

    /// Creates a layer normalization layer.
    ///
    /// - Parameters:
    ///   - featureCount: The number of features.
    ///   - axis: The axis that should be normalized.
    ///   - epsilon: The small scalar added to variance.
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

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    ///   - context: The contextual information for the layer application, e.g. the current learning
    ///     phase.
    /// - Returns: The output.
    @differentiable
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        let mean = input.mean(alongAxes: axis)
        let variance = input.variance(alongAxes: axis)
        let inv = rsqrt(variance + epsilon) * scale
        return (input - mean) * inv + offset
    }
}

public extension Tensor where Scalar: TensorFlowFloatingPoint {
    /// Computes dropout given a probability.
    @differentiable(wrt: self where Scalar: Differentiable)
    func droppingOut(probability: Double) -> Tensor {
        let noise = Tensor(randomUniform: shape)
        let keepMask = noise .>= Scalar(probability)
        let keepProbability = Scalar(1.0 - probability)
        return self * Tensor(keepMask) / Tensor(keepProbability)
    }
}

/// A dropout layer.
///
/// Dropout consists in randomly setting a fraction of input units to `0` at each update during
/// training time, which helps prevent overfitting.
@_fixed_layout
public struct Dropout<Scalar: TensorFlowFloatingPoint>: Layer {
    @noDerivative public let probability: Double

    /// Creates a dropout layer.
    ///
    /// - Parameter probability: The drop probability.
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

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    ///   - context: The contextual information for the layer application, e.g. the current learning
    ///     phase.
    /// - Returns: The output.
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

/// An upsampling layer for 1-D inputs.
@_fixed_layout
public struct UpSampling1D<Scalar: TensorFlowFloatingPoint>: Layer {
    @noDerivative public let size: Int32

    /// Creates an upsampling layer.
    ///
    /// - Parameter size: The upsampling factor for timesteps.
    public init(size: Int32) {
       self.size = size
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    ///   - context: The contextual information for the layer application, e.g. the current learning
    ///     phase.
    /// - Returns: The output.
    @differentiable
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        let shape = input.shape
        let (batchSize, timesteps, channels) = (shape[0], shape[1], shape[2])
        let scaleOnes = Tensor<Scalar>(ones: [1, 1, size, 1])
        let upSampling = input.reshaped(to: [batchSize, timesteps, 1, channels]) * scaleOnes
        return upSampling.reshaped(to: [batchSize, timesteps * size, channels])
    }
}

/// An upsampling layer for 2-D inputs.
@_fixed_layout
public struct UpSampling2D<Scalar: TensorFlowFloatingPoint>: Layer {
    @noDerivative public let size: Int32

    /// Creates an upsampling layer.
    ///
    /// - Parameter size: The upsampling factor for rows and columns.
    public init(size: Int32) {
       self.size = size
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    ///   - context: The contextual information for the layer application, e.g. the current learning
    ///     phase.
    /// - Returns: The output.
    @differentiable
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        let shape = input.shape
        let (batchSize, height, width, channels) = (shape[0], shape[1], shape[2], shape[3])
        let scaleOnes = Tensor<Scalar>(ones: [1, 1, size, 1, size, 1])
        let upSampling = input.reshaped(to: [batchSize, height, 1, width, 1, channels]) * scaleOnes
        return upSampling.reshaped(to: [batchSize, height * size, width * size, channels])
    }
}

/// A flatten layer.
///
/// A flatten layer flattens the input when applied without affecting the batch size.
@_fixed_layout
public struct Flatten<Scalar: TensorFlowFloatingPoint>: Layer {
    /// Creates a flatten layer.
    public init() {}

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    ///   - context: The contextual information for the layer application, e.g. the current learning
    ///     phase.
    /// - Returns: The output.
    @differentiable
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        let batchSize = input.shape[0]
        let remaining = input.shape[1..<input.rank].contiguousSize
        return input.reshaped(to: [batchSize, remaining])
    }
}

/// A reshape layer.
@_fixed_layout
public struct Reshape<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The target shape.
    @noDerivative public let shape: Tensor<Int32>

    // TF-331 workaround:
    @usableFromInline
    internal var _nontrivial = Tensor<Float>(0)

    /// Creates a reshape layer.
    ///
    /// - Parameter shape: The target shape, represented by a tensor.
    public init(shape: Tensor<Int32>) {
        self.shape = shape
    }

    /// Creates a reshape layer.
    ///
    /// - Parameter shape: The target shape.
    public init(_ shape: TensorShape) {
        self.init(shape: Tensor(shape.dimensions))
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    ///   - context: The contextual information for the layer application, e.g. the current learning
    ///     phase.
    /// - Returns: The output.
    @differentiable
    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
        return input.reshaped(toShape: shape)
    }
}

/// An input to a recurrent neural network.
public struct RNNInput<TimeStepInput: Differentiable, State: Differentiable>: Differentiable {
    /// The input at the current time step.
    public var timeStepInput: TimeStepInput
    /// The previous state.
    public var previousState: State

    @differentiable
    public init(timeStepInput: StepInput, previousState: State) {
        self.timeStepInput = timeStepInput
        self.previousState = previousState
    }
}

/// A recurrent neural network cell.
public protocol RNNCell: Layer where Input == RNNInput<TimeStepInput, State> {
    /// The input at a time step.
    associatedtype TimeStepInput: Differentiable
    /// The state that may be preserved across time steps.
    typealias State = Output
    /// The zero state.
    var zeroState: State { get }
}

public extension RNNCell {
    /// Returns the new state obtained from applying the RNN cell to the input at the current time
    /// step and the previous state.
    ///
    /// - Parameters:
    ///   - timeStepInput: The input at the current time step.
    ///   - previousState: The previous state of the RNN cell.
    ///   - context: The contextual information for the layer application, e.g. the current learning
    ///     phase.
    /// - Returns: The output.
    @differentiable
    func applied(to timeStepInput: StepInput, previous: State, in context: Context) -> State {
        return applied(to: Input(timeStepInput: timeStepInput, previousState: previous),
                       in: context)
    }
}
