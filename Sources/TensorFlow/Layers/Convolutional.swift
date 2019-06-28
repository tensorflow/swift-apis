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

/// A 1-D convolution layer (e.g. temporal convolution over a time-series).
///
/// This layer creates a convolution filter that is convolved with the layer input to produce a
/// tensor of outputs.
@frozen
public struct Conv1D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The 3-D convolution kernel `[width, inputChannels, outputChannels]`.
    public var filter: Tensor<Scalar>
    /// The bias vector `[outputChannels]`.
    public var bias: Tensor<Scalar>
    /// An activation function.
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
    /// The element-wise activation function.
    @noDerivative public let activation: Activation
    /// The stride of the sliding window for the temporal dimension.
    @noDerivative public let stride: Int
    /// The padding algorithm for convolution.
    @noDerivative public let padding: Padding
    /// The dilation factor for the temporal dimension.
    @noDerivative public let dilation: Int

    /// Creates a `Conv1D` layer with the specified filter, bias, activation function, stride,
    /// dilation and padding.
    ///
    /// - Parameters:
    ///   - filter: The 3-D convolution kernel `[width, inputChannels, outputChannels]`.
    ///   - bias: The bias vector `[outputChannels]`.
    ///   - activation: The element-wise activation function.
    ///   - stride: The stride of the sliding window for the temporal dimension.
    ///   - padding: The padding algorithm for convolution.
    ///   - dilation: The dilation factor for the temporal dimension.
    public init(
        filter: Tensor<Scalar>,
        bias: Tensor<Scalar>,
        activation: @escaping Activation = identity,
        stride: Int = 1,
        padding: Padding = .valid,
        dilation: Int = 1
    ) {
        self.filter = filter
        self.bias = bias
        self.activation = activation
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer `[batchCount, width, inputChannels]`.
    /// - Returns: The output `[batchCount, newWidth, outputChannels]`.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let conv = conv2D(
            input.expandingShape(at: 1),
            filter: filter.expandingShape(at: 0),
            strides: (1, 1, stride, 1),
            padding: padding,
            dilations: (1, 1, dilation, 1))
        return activation(conv.squeezingShape(at: 1) + bias)
    }
}

public extension Conv1D where Scalar.RawSignificand: FixedWidthInteger {
    /// Creates a `Conv1D` layer with the specified filter shape, stride, padding, dilation and
    /// element-wise activation function. The filter tensor is initialized using Glorot uniform
    /// initialization with the specified generator. The bias vector is initialized with zeros.
    ///
    /// - Parameters:
    ///   - filterShape: The 3-D shape of the filter, representing
    ///     `[width, inputChannels, outputChannels]`.
    ///   - stride: The stride of the sliding window for the temporal dimension.
    ///   - padding: The padding algorithm for convolution.
    ///   - dilation: The dilation factor for the temporal dimension.
    ///   - activation: The element-wise activation function.
    ///   - generator: The random number generator for initialization.
    ///
    /// - Note: Use `init(filterShape:stride:padding:dilation:activation:seed:)` for faster random
    ///   initialization.
    init<G: RandomNumberGenerator>(
        filterShape: (Int, Int, Int),
        stride: Int = 1,
        padding: Padding = .valid,
        dilation: Int = 1,
        activation: @escaping Activation = identity,
        generator: inout G
    ) {
        let filterTensorShape = TensorShape([
            filterShape.0, filterShape.1, filterShape.2])
        self.init(
            filter: Tensor(glorotUniform: filterTensorShape),
            bias: Tensor(zeros: [filterShape.2]),
            activation: activation,
            stride: stride,
            padding: padding,
            dilation: dilation)
    }
}

public extension Conv1D {
    /// Creates a `Conv1D` layer with the specified filter shape, strides, padding, dilation and
    /// element-wise activation function. The filter tensor is initialized using Glorot uniform
    /// initialization with the specified seed. The bias vector is initialized with zeros.
    ///
    /// - Parameters:
    ///   - filterShape: The 3-D shape of the filter, representing
    ///     `[width, inputChannels, outputChannels]`.
    ///   - stride: The stride of the sliding window for the temporal dimension.
    ///   - padding: The padding algorithm for convolution.
    ///   - dilation: The dilation factor for the temporal dimension.
    ///   - activation: The element-wise activation function.
    ///   - seed: The random seed for initialization. The default value is random.
    init(
        filterShape: (Int, Int, Int),
        stride: Int = 1,
        padding: Padding = .valid,
        dilation: Int = 1,
        activation: @escaping Activation = identity,
        seed: (Int32, Int32) = (Int32.random(in: Int32.min..<Int32.max),
                                Int32.random(in: Int32.min..<Int32.max))
    ) {
        let filterTensorShape = TensorShape([
            filterShape.0, filterShape.1, filterShape.2])
        self.init(
            filter: Tensor(glorotUniform: filterTensorShape, seed: seed),
            bias: Tensor(zeros: [filterShape.2]),
            activation: activation,
            stride: stride,
            padding: padding,
            dilation: dilation)
    }
}

/// A 2-D convolution layer (e.g. spatial convolution over images).
///
/// This layer creates a convolution filter that is convolved with the layer input to produce a
/// tensor of outputs.
@frozen
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
    @noDerivative public let strides: (Int, Int)
    /// The padding algorithm for convolution.
    @noDerivative public let padding: Padding
    /// The dilation factor for spatial dimensions.
    @noDerivative public let dilations: (Int, Int)

    /// Creates a `Conv2D` layer with the specified filter, bias, activation function, strides,
    /// dilations and padding.
    ///
    /// - Parameters:
    ///   - filter: The 4-D convolution kernel.
    ///   - bias: The bias vector.
    ///   - activation: The element-wise activation function.
    ///   - strides: The strides of the sliding window for spatial dimensions.
    ///   - padding: The padding algorithm for convolution.
    ///   - dilations: The dilation factor for spatial dimensions.
    public init(
        filter: Tensor<Scalar>,
        bias: Tensor<Scalar>,
        activation: @escaping Activation = identity,
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        dilations: (Int, Int) = (1, 1)
    ) {
        self.filter = filter
        self.bias = bias
        self.activation = activation
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return activation(conv2D(
            input,
            filter: filter,
            strides: (1, strides.0, strides.1, 1),
            padding: padding,
            dilations: (1, dilations.0, dilations.1, 1)) + bias)
    }
}

public extension Conv2D {
    /// Creates a `Conv2D` layer with the specified filter shape, strides, padding, dilations and
    /// element-wise activation function. The filter tensor is initialized using Glorot uniform
    /// initialization with the specified generator. The bias vector is initialized with zeros.
    ///
    /// - Parameters:
    ///   - filterShape: The shape of the 4-D convolution kernel.
    ///   - strides: The strides of the sliding window for spatial dimensions.
    ///   - padding: The padding algorithm for convolution.
    ///   - dilations: The dilation factor for spatial dimensions.
    ///   - activation: The element-wise activation function.
    ///   - generator: The random number generator for initialization.
    ///
    /// - Note: Use `init(filterShape:strides:padding:activation:seed:)` for faster random
    ///   initialization.
    init<G: RandomNumberGenerator>(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        dilations: (Int, Int) = (1, 1),
        activation: @escaping Activation = identity,
        generator: inout G
    ) {
        let filterTensorShape = TensorShape([
            filterShape.0, filterShape.1, filterShape.2, filterShape.3])
        self.init(
            filter: Tensor(glorotUniform: filterTensorShape, generator: &generator),
            bias: Tensor(zeros: [filterShape.3]),
            activation: activation,
            strides: strides,
            padding: padding,
            dilations: dilations)
    }
}

public extension Conv2D {
    /// Creates a `Conv2D` layer with the specified filter shape, strides, padding, dilations and
    /// element-wise activation function. The filter tensor is initialized using Glorot uniform
    /// initialization with the specified seed. The bias vector is initialized with zeros.
    ///
    /// - Parameters:
    ///   - filterShape: The shape of the 4-D convolution kernel.
    ///   - strides: The strides of the sliding window for spatial dimensions.
    ///   - padding: The padding algorithm for convolution.
    ///   - dilations: The dilation factor for spatial dimensions.
    ///   - activation: The element-wise activation function.
    ///   - seed: The random seed for initialization. The default value is random.
    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        dilations: (Int, Int) = (1, 1),
        activation: @escaping Activation = identity,
        seed: (Int32, Int32) = (Int32.random(in: Int32.min..<Int32.max),
                                Int32.random(in: Int32.min..<Int32.max))
    ) {
        let filterTensorShape = TensorShape([
            filterShape.0, filterShape.1, filterShape.2, filterShape.3])
        self.init(
            filter: Tensor(glorotUniform: filterTensorShape, seed: seed),
            bias: Tensor(zeros: [filterShape.3]),
            activation: activation,
            strides: strides,
            padding: padding,
            dilations: dilations)
    }
}

/// A 3-D convolution layer for spatial/spatio-temporal convolution over images.
///
/// This layer creates a convolution filter that is convolved with the layer input to produce a
/// tensor of outputs.
@frozen
public struct Conv3D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The 5-D convolution kernel.
    public var filter: Tensor<Scalar>
    /// The bias vector.
    public var bias: Tensor<Scalar>
    /// An activation function.
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
    /// The element-wise activation function.
    @noDerivative public let activation: Activation
    /// The strides of the sliding window for spatial dimensions.
    @noDerivative public let strides: (Int, Int, Int)
    /// The padding algorithm for convolution.
    @noDerivative public let padding: Padding

    /// Creates a `Conv3D` layer with the specified filter, bias, activation function, strides, and
    /// padding.
    ///
    /// - Parameters:
    ///   - filter: The 5-D convolution kernel.
    ///   - bias: The bias vector.
    ///   - activation: The element-wise activation function.
    ///   - strides: The strides of the sliding window for spatial dimensions.
    ///   - padding: The padding algorithm for convolution.
    public init(
        filter: Tensor<Scalar>,
        bias: Tensor<Scalar>,
        activation: @escaping Activation = identity,
        strides: (Int, Int, Int) = (1, 1, 1),
        padding: Padding = .valid
    ) {
        self.filter = filter
        self.bias = bias
        self.activation = activation
        self.strides = strides
        self.padding = padding
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return activation(conv3D(
            input,
            filter: filter,
            strides: (1, strides.0, strides.1, strides.2, 1),
            padding: padding) + bias)
    }
}

public extension Conv3D {
    /// Creates a `Conv3D` layer with the specified filter shape, strides, padding, and
    /// element-wise activation function. The filter tensor is initialized using Glorot uniform
    /// initialization with the specified generator. The bias vector is initialized with zeros.
    ///
    /// - Parameters:
    ///   - filterShape: The shape of the 5-D convolution kernel.
    ///   - strides: The strides of the sliding window for spatial/spatio-temporal dimensions.
    ///   - padding: The padding algorithm for convolution.
    ///   - activation: The element-wise activation function.
    ///   - generator: The random number generator for initialization.
    ///
    /// - Note: Use `init(filterShape:strides:padding:activation:seed:)` for faster random
    ///   initialization.
    init<G: RandomNumberGenerator>(
        filterShape: (Int, Int, Int, Int, Int),
        strides: (Int, Int, Int) = (1, 1, 1),
        padding: Padding = .valid,
        activation: @escaping Activation = identity,
        generator: inout G
    ) {
        let filterTensorShape = TensorShape([
            filterShape.0, filterShape.1, filterShape.2, filterShape.3, filterShape.4])
        self.init(
            filter: Tensor(glorotUniform: filterTensorShape, generator: &generator),
            bias: Tensor(zeros: [filterShape.4]),
            activation: activation,
            strides: strides,
            padding: padding)
    }
}

public extension Conv3D {
    /// Creates a `Conv3D` layer with the specified filter shape, strides, padding, and
    /// element-wise activation function. The filter tensor is initialized using Glorot uniform
    /// initialization with the specified seed. The bias vector is initialized with zeros.
    ///
    /// - Parameters:
    ///   - filterShape: The shape of the 5-D convolution kernel.
    ///   - strides: The strides of the sliding window for spatial/spatio-temporal dimensions.
    ///   - padding: The padding algorithm for convolution.
    ///   - activation: The element-wise activation function.
    ///   - seed: The random seed for initialization. The default value is random.
    init(
        filterShape: (Int, Int, Int, Int, Int),
        strides: (Int, Int, Int) = (1, 1, 1),
        padding: Padding = .valid,
        activation: @escaping Activation = identity,
        seed: (Int32, Int32) = (Int32.random(in: Int32.min..<Int32.max),
                                Int32.random(in: Int32.min..<Int32.max))
    ) {
        let filterTensorShape = TensorShape([
            filterShape.0, filterShape.1, filterShape.2, filterShape.3, filterShape.4])
        self.init(
            filter: Tensor(glorotUniform: filterTensorShape, seed: seed),
            bias: Tensor(zeros: [filterShape.4]),
            activation: activation,
            strides: strides,
            padding: padding)
    }
}

/// A 2-D transposed convolution layer (e.g. spatial transposed convolution over images).
///
/// This layer creates a convolution filter that is transpose-convolved with the layer input
/// to produce a tensor of outputs.
@frozen
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
    @noDerivative public let strides: (Int, Int)
    /// The padding algorithm for convolution.
    @noDerivative public let padding: Padding
    @noDerivative public let paddingIndex: Int

    /// Creates a `TransposedConv2D` layer with the specified filter, bias,
    /// activation function, strides, and padding.
    ///
    /// - Parameters:
    ///   - filter: A 4-D tensor of shape
    ///     `[width, height, input channel count, output channel count]`.
    ///   - bias: The bias tensor of shape `[output channel count]`.
    ///   - activation: The element-wise activation function.
    ///   - strides: The strides of the sliding window for spatial dimensions.
    ///   - padding: The padding algorithm for convolution.
    public init(
        filter: Tensor<Float>,
        bias: Tensor<Float>,
        activation: @escaping Activation = identity,
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid
    ) {
        self.filter = filter
        self.bias = bias
        self.activation = activation
        self.strides = strides
        self.padding = padding
        self.paddingIndex = padding == .same ? 0 : 1
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let batchSize = input.shape[0]
        let w = (input.shape[1] - (1 * paddingIndex)) *
          strides.0 + (filter.shape[0] * paddingIndex)
        let h = (input.shape[2] - (1 * paddingIndex)) *
          strides.1 + (filter.shape[1] * paddingIndex)
        let c = filter.shape[2]
        let newShape = Tensor<Int32>([Int32(batchSize), Int32(w), Int32(h), Int32(c)])
        return activation(conv2DBackpropInput(
            input,
            shape: newShape,
            filter: filter,
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
    ///   - filterShape: A 4-D tensor of shape
    ///     `[width, height, input channel count, output channel count]`.
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
            filterShape.0, filterShape.1, filterShape.2, filterShape.3])
        self.init(
            filter: Tensor(glorotUniform: filterTensorShape, generator: &generator),
            bias: Tensor(zeros: [filterShape.3]),
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
    ///   - filterShape: A 4-D tensor of shape
    ///     `[width, height, input channel count, output channel count]`.
    ///   - strides: The strides of the sliding window for spatial dimensions.
    ///   - padding: The padding algorithm for convolution.
    ///   - activation: The element-wise activation function.
    ///   - seed: The random seed for initialization. The default value is random.
    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        activation: @escaping Activation = identity,
        seed: (Int32, Int32) = (Int32.random(in: Int32.min..<Int32.max),
                                Int32.random(in: Int32.min..<Int32.max))
    ) {
        let filterTensorShape = TensorShape([
            filterShape.0, filterShape.1, filterShape.2, filterShape.3])
        self.init(
            filter: Tensor(glorotUniform: filterTensorShape, seed: seed),
            bias: Tensor(zeros: [filterShape.3]),
            activation: activation,
            strides: strides,
            padding: padding)
    }
}

/// A 2-D depthwise convolution layer.
///
/// This layer creates seperable convolution filters that are convolved with the layer input to produce a
/// tensor of outputs.
@frozen
public struct DepthwiseConv2D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The 4-D convolution kernel.
    public var filter: Tensor<Scalar>
    /// The bias vector.
    public var bias: Tensor<Scalar>
    /// An activation function.
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
    /// The element-wise activation function.
    @noDerivative public let activation: Activation
    /// The strides of the sliding window for spatial dimensions.
    @noDerivative public let strides: (Int, Int)
    /// The padding algorithm for convolution.
    @noDerivative public let padding: Padding

    /// Creates a `DepthwiseConv2D` layer with the specified filter, bias, activation function, strides, and
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
        activation: @escaping Activation = identity,
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid
    ) {
        self.filter = filter
        self.bias = bias
        self.activation = activation
        self.strides = strides
        self.padding = padding
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return activation(depthwiseConv2D(
            input,
            filter: filter,
            strides: (1, strides.0, strides.1, 1),
            padding: padding) + bias)
    }
}

public extension DepthwiseConv2D {
    /// Creates a `DepthwiseConv2D` layer with the specified filter shape, strides, padding, and
    /// element-wise activation function. The filter tensor is initialized using Glorot uniform
    /// initialization with the specified generator. The bias vector is initialized with zeros.
    ///
    /// - Parameters:
    ///   - filterShape: The shape of the 4-D convolution kernel.
    ///   - strides: The strides of the sliding window for spatial/spatio-temporal dimensions.
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
            filterShape.0, filterShape.1, filterShape.2, filterShape.3])
        self.init(
            filter: Tensor(glorotUniform: filterTensorShape, generator: &generator),
            bias: Tensor(zeros: [filterShape.3]),
            activation: activation,
            strides: strides,
            padding: padding)
    }
}

public extension DepthwiseConv2D {
    /// Creates a `depthwiseConv2D` layer with the specified filter shape, strides, padding, and
    /// element-wise activation function. The filter tensor is initialized using Glorot uniform
    /// initialization with the specified seed. The bias vector is initialized with zeros.
    ///
    /// - Parameters:
    ///   - filterShape: The shape of the 4-D convolution kernel.
    ///   - strides: The strides of the sliding window for spatial/spatio-temporal dimensions.
    ///   - padding: The padding algorithm for convolution.
    ///   - activation: The element-wise activation function.
    ///   - seed: The random seed for initialization. The default value is random.
    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        activation: @escaping Activation = identity,
        seed: (Int32, Int32) = (Int32.random(in: Int32.min..<Int32.max),
                                Int32.random(in: Int32.min..<Int32.max))
    ) {
        let filterTensorShape = TensorShape([
            filterShape.0, filterShape.1, filterShape.2, filterShape.3])
        self.init(
            filter: Tensor(glorotUniform: filterTensorShape, seed: seed),
            bias: Tensor(zeros: [filterShape.3]),
            activation: activation,
            strides: strides,
            padding: padding)
    }
}

/// A layer for cropping tensors along the temporal dimension.
public struct Cropping1D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The cropping dimensions along the temporal axis.
    @noDerivative public let cropping: (Int, Int)

    /// Creates a cropping layer to trim the temporal dimension.
    ///
    /// - Parameter cropping: A tuple of two integers describing how many units should be trimmed
    ///   off the beginning and end of the cropping dimension.
    public init(cropping: (Int, Int) = (1, 1)) {
        self.cropping = cropping
    }

    /// Creates a cropping layer to trim the temporal dimension.
    ///
    /// - Parameter cropping: An integer describing how many units should be symmetrically trimmed
    ///   off the beginning and end of the cropping dimension. i.e. `(cropping, cropping)`.
    public init(cropping: Int = 1) {
        self.cropping = (cropping, cropping)
    }

    /// Returns the cropped input tensor according to the `cropping` dimensions specified
    /// at initialization.
    ///
    /// - Parameter input: The 3D tensor to be cropped. Note the expected shape of the input is
    ///   `[batch size, axis to crop, features]`.
    /// - Returns: The cropped 3D tensor of shape `[batch, cropped axis, feature]`.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let (begin, end) = cropping
        return input.slice(
            lowerBounds: [0, begin, 0],
            upperBounds: [input.shape[0], input.shape[1] - end, input.shape[2]])
    }
}

/// A layer for cropping tensors along spatial dimensions, e.g. image cropping.
///
/// `Cropping2D` can trim an input at the top, bottom, left, and right sides of
/// its spatial dimensions.
public struct Cropping2D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The cropping dimensions along the height and width axes.
    @noDerivative public let cropping: ((Int, Int), (Int, Int))

    /// Creates a cropping layer to trim spatial dimensions, i.e. height and width.
    ///
    /// - Parameter cropping: A tuple of two tuples containing two integers describing how many
    ///   units should be trimmed off the height and width dimensions,
    ///   i.e. `((top, bottom), (left, right))`.
    // TODO: Add data format property support when control flow differentiation is completed.
    public init(cropping: ((Int, Int), (Int, Int)) = ((0, 0), (0, 0))) {
        self.cropping = cropping
    }

    /// Creates a cropping layer to trim spatial dimensions, i.e. height and width.
    ///
    /// - Parameter cropping: A tuple of two integers describing how many units should be
    ///   symmetrically trimmed off the height and width dimensions,
    ///   i.e. `((height, height), (width, width))`.
    public init(cropping: (Int, Int) = (0, 0)) {
        let (height, width) = cropping
        self.init(cropping: ((height, height), (width, width)))
    }

    /// Returns the cropped input tensor according to the `cropping` dimensions specified
    /// at initialization.
    ///
    /// - Parameter input: The 4D tensor to be cropped. Note that the expected data format is 
    ///   channel-last, i.e. shape `[batch size, height, width, channel count]`.
    /// - Returns: The cropped tensor.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let (top, bottom) = cropping.0
        let (left, right) = cropping.1
        return input.slice(
            lowerBounds: [0, top, left, 0],
            upperBounds: [input.shape[0], input.shape[1] - bottom,
                input.shape[2] - right, input.shape[3]])
    }
}
