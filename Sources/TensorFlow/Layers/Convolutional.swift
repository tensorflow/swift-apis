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
    /// The 3-D convolution filter.
    public var filter: Tensor<Scalar>
    /// The bias vector.
    public var bias: Tensor<Scalar>
    /// The element-wise activation function.
    @noDerivative public let activation: Activation
    /// The stride of the sliding window for the temporal dimension.
    @noDerivative public let stride: Int
    /// The padding algorithm for convolution.
    @noDerivative public let padding: Padding
    /// The dilation factor for the temporal dimension.
    @noDerivative public let dilation: Int

    /// The element-wise activation function type.
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

    /// Creates a `Conv1D` layer with the specified filter, bias, activation function, stride,
    /// dilation and padding.
    ///
    /// - Parameters:
    ///   - filter: The 3-D convolution filter of shape
    ///     [filter width, input channel count, output channel count].
    ///   - bias: The bias vector of shape [output channel count].
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
    /// The output width is computed as:
    ///
    /// output width =
    /// (input width + 2 * padding size - (dilation * (filter width - 1) + 1)) / stride + 1
    ///
    /// and padding size is determined by the padding scheme.
    ///
    /// - Parameter input: The input to the layer [batch size, input width, input channel count].
    /// - Returns: The output of shape [batch size, output width, output channel count].
    ///
    /// - Note: Padding size equals zero when using `.valid`.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        activation(conv1D(
            input,
            filter: filter,
            stride: stride,
            padding: padding,
            dilation: dilation) + bias)
    }
}

public extension Conv1D where Scalar.RawSignificand: FixedWidthInteger {
    /// Creates a `Conv1D` layer with the specified filter shape, stride, padding, dilation and
    /// element-wise activation function.
    ///
    /// - Parameters:
    ///   - filterShape: The 3-D shape of the filter, representing
    ///     (filter width, input channel count, output channel count).
    ///   - stride: The stride of the sliding window for the temporal dimension.
    ///   - padding: The padding algorithm for convolution.
    ///   - dilation: The dilation factor for the temporal dimension.
    ///   - activation: The element-wise activation function.
    ///   - filterInitializer: Initializer to use for the filter parameters.
    ///   - biasInitializer: Initializer to use for the bias parameters.
    init(
        filterShape: (Int, Int, Int),
        stride: Int = 1,
        padding: Padding = .valid,
        dilation: Int = 1,
        activation: @escaping Activation = identity,
        filterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
        biasInitializer: ParameterInitializer<Scalar> = zeros()
    ) {
        let filterTensorShape = TensorShape([
            filterShape.0, filterShape.1, filterShape.2])
        self.init(
            filter: filterInitializer(filterTensorShape),
            bias: biasInitializer([filterShape.2]),
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
    /// The 4-D convolution filter.
    public var filter: Tensor<Scalar>
    /// The bias vector.
    public var bias: Tensor<Scalar>
    /// The element-wise activation function.
    @noDerivative public let activation: Activation
    /// The strides of the sliding window for spatial dimensions.
    @noDerivative public let strides: (Int, Int)
    /// The padding algorithm for convolution.
    @noDerivative public let padding: Padding
    /// The dilation factor for spatial dimensions.
    @noDerivative public let dilations: (Int, Int)

    /// The element-wise activation function type.
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

    /// Creates a `Conv2D` layer with the specified filter, bias, activation function, strides,
    /// dilations and padding.
    ///
    /// - Parameters:
    ///   - filter: The 4-D convolution filter of shape
    ///     [filter height, filter width, input channel count, output channel count].
    ///   - bias: The bias vector of shape [output channel count].
    ///   - activation: The element-wise activation function.
    ///   - strides: The strides of the sliding window for spatial dimensions, i.e.
    ///     (stride height, stride width).
    ///   - padding: The padding algorithm for convolution.
    ///   - dilations: The dilation factors for spatial dimensions, i.e.
    ///     (dilation height, dilation width).
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
    /// The output spatial dimensions are computed as:
    ///
    /// output height =
    /// (input height + 2 * padding height - (dilation height * (filter height - 1) + 1))
    /// / stride height + 1
    ///
    /// output width =
    /// (input width + 2 * padding width - (dilation width * (filter width - 1) + 1))
    /// / stride width + 1
    ///
    /// and padding sizes are determined by the padding scheme.
    ///
    /// - Parameter input: The input to the layer of shape
    ///   [batch size, input height, input width, input channel count].
    /// - Returns: The output of shape
    ///   [batch count, output height, output width, output channel count].
    ///
    /// - Note: Padding size equals zero when using `.valid`.
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
    /// element-wise activation function.
    ///
    /// - Parameters:
    ///   - filterShape: The shape of the 4-D convolution filter, representing
    ///     (filter height, filter width, input channel count, output channel count).
    ///   - strides: The strides of the sliding window for spatial dimensions, i.e.
    ///     (stride height, stride width).
    ///   - padding: The padding algorithm for convolution.
    ///   - dilations: The dilation factors for spatial dimensions, i.e.
    ///     (dilation height, dilation width).
    ///   - activation: The element-wise activation function.
    ///   - filterInitializer: Initializer to use for the filter parameters.
    ///   - biasInitializer: Initializer to use for the bias parameters.
    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        dilations: (Int, Int) = (1, 1),
        activation: @escaping Activation = identity,
        filterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
        biasInitializer: ParameterInitializer<Scalar> = zeros()
    ) {
        let filterTensorShape = TensorShape([
            filterShape.0, filterShape.1, filterShape.2, filterShape.3])
        self.init(
            filter: filterInitializer(filterTensorShape),
            bias: biasInitializer([filterShape.3]),
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
    /// The 5-D convolution filter.
    public var filter: Tensor<Scalar>
    /// The bias vector.
    public var bias: Tensor<Scalar>
    /// The element-wise activation function.
    @noDerivative public let activation: Activation
    /// The strides of the sliding window for spatial dimensions.
    @noDerivative public let strides: (Int, Int, Int)
    /// The padding algorithm for convolution.
    @noDerivative public let padding: Padding
    /// The dilation factor for spatial/spatio temporal dimensions.
    @noDerivative public let dilations: (Int, Int, Int)

    /// The element-wise activation function type.
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

    /// Creates a `Conv3D` layer with the specified filter, bias, activation function, strides, and
    /// padding.
    ///
    /// - Parameters:
    ///   - filter: The 5-D convolution filter of shape
    ///     [filter depth, filter height, filter width, input channel count,
    ///     output channel count].
    ///   - bias: The bias vector of shape [output channel count].
    ///   - activation: The element-wise activation function.
    ///   - strides: The strides of the sliding window for spatial dimensions, i.e.
    ///     (stride depth, stride height, stride width)
    ///   - padding: The padding algorithm for convolution.
    ///   - dilations: The dilation factor for spatial/spatio-temporal dimensions.
    public init(
        filter: Tensor<Scalar>,
        bias: Tensor<Scalar>,
        activation: @escaping Activation = identity,
        strides: (Int, Int, Int) = (1, 1, 1),
        padding: Padding = .valid,
        dilations: (Int, Int, Int) = (1, 1, 1)
    ) {
        precondition(dilations.2 == 1,
                     "Dilations in the depth dimension must be 1.")
        self.filter = filter
        self.bias = bias
        self.activation = activation
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// The output spatial dimensions are computed as:
    ///
    /// output depth =
    /// (input depth + 2 * padding depth - (dilation depth * (filter depth - 1) + 1))
    /// / stride depth + 1
    ///
    /// output height =
    /// (input height + 2 * padding height - (dilation height * (filter height - 1) + 1))
    /// / stride height + 1
    ///
    /// output width =
    /// (input width + 2 * padding width - (dilation width * (filter width - 1) + 1))
    /// / stride width + 1
    ///
    /// and padding sizes are determined by the padding scheme.
    ///
    /// - Parameter input: The input to the layer of shape
    ///   [batch count, input depth, input height, input width, input channel count].
    /// - Returns: The output of shape
    ///   [batch count, output depth, output height, output width, output channel count].
    ///
    /// - Note: Padding size equals zero when using `.valid`.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return activation(conv3D(
            input,
            filter: filter,
            strides: (1, strides.0, strides.1, strides.2, 1),
            padding: padding,
            dilations: (1, dilations.0, dilations.1, dilations.2, 1)
        ) + bias)
    }
}

public extension Conv3D {
    /// Creates a `Conv3D` layer with the specified filter shape, strides, padding, dilations and
    /// element-wise activation function. The filter tensor is initialized using Glorot uniform
    /// initialization with the specified seed. The bias vector is initialized with zeros.
    ///
    /// - Parameters:
    ///   - filterShape: The shape of the 5-D convolution filter, representing
    ///     (filter depth, filter height, filter width, input channel count,
    ///     output channel count).
    ///   - strides: The strides of the sliding window for spatial dimensions, i.e.
    ///     (stride depth, stride height, stride width)
    ///   - padding: The padding algorithm for convolution.
    ///   - dilations: The dilation factor for spatial/spatio-temporal dimensions.
    ///   - activation: The element-wise activation function.
    ///   - filterInitializer: Initializer to use for the filter parameters.
    ///   - biasInitializer: Initializer to use for the bias parameters.
    init(
        filterShape: (Int, Int, Int, Int, Int),
        strides: (Int, Int, Int) = (1, 1, 1),
        padding: Padding = .valid,
        dilations: (Int, Int, Int) = (1, 1, 1),
        activation: @escaping Activation = identity,
        filterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
        biasInitializer: ParameterInitializer<Scalar> = zeros()
    ) {
        let filterTensorShape = TensorShape([
            filterShape.0, filterShape.1, filterShape.2, filterShape.3, filterShape.4])
        self.init(
            filter: filterInitializer(filterTensorShape),
            bias: biasInitializer([filterShape.4]),
            activation: activation,
            strides: strides,
            padding: padding,
            dilations: dilations)
    }
}

/// A 1-D transposed convolution layer (e.g. temporal transposed convolution over images).
///
/// This layer creates a convolution filter that is transpose-convolved with the layer input
/// to produce a tensor of outputs.
@frozen
public struct TransposedConv1D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The 1-D convolution kernel.
    public var filter: Tensor<Scalar>
    /// The bias vector.
    public var bias: Tensor<Scalar>
    /// The element-wise activation function.
    @noDerivative public let activation: Activation
    /// The strides of the sliding window for spatial dimensions.
    @noDerivative public let stride: Int
    /// The padding algorithm for convolution.
    @noDerivative public let padding: Padding
    /// The paddingIndex property allows us to handle computation based on padding.
    @noDerivative public let paddingIndex: Int

    /// The element-wise activation function type.
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

    /// Creates a `TransposedConv1D` layer with the specified filter, bias,
    /// activation function, strides, and padding.
    ///
    /// - Parameters:
    ///   - filter: The 3-D convolution kernel.
    ///   - bias: The bias vector.
    ///   - activation: The element-wise activation function.
    ///   - strides: The strides of the sliding window for spatial dimensions.
    ///   - padding: The padding algorithm for convolution.
    public init(
        filter: Tensor<Scalar>,
        bias: Tensor<Scalar>,
        activation: @escaping Activation = identity,
        stride: Int = 1,
        padding: Padding = .valid
    ) {
        self.filter = filter
        self.bias = bias
        self.activation = activation
        self.stride = stride
        self.padding = padding
        self.paddingIndex = padding == .same ? 0 : 1
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let batchSize = input.shape[0]
        let w = (input.shape[1] - (1 * paddingIndex)) *
            stride + (filter.shape[0] * paddingIndex)
        let c = filter.shape[2]
        let newShape = Tensor<Int32>([Int32(batchSize), 1, Int32(w), Int32(c)])
        return activation(conv2DBackpropInput(
            input.expandingShape(at: 1),
            shape: newShape,
            filter: filter.expandingShape(at: 0),
            strides: (1, 1, stride, 1),
            padding: padding) + bias)
    }
}

public extension TransposedConv1D {
    /// Creates a `TransposedConv1D` layer with the specified filter shape, strides, padding, and
    /// element-wise activation function. The filter tensor is initialized using Glorot uniform
    /// initialization with the specified generator. The bias vector is initialized with zeros.
    ///
    /// - Parameters:
    ///   - filterShape: The shape of the 3-D convolution kernel.
    ///   - strides: The strides of the sliding window for spatial dimensions.
    ///   - padding: The padding algorithm for convolution.
    ///   - activation: The element-wise activation function.
    ///   - generator: The random number generator for initialization.
    init(
        filterShape: (Int, Int, Int),
        stride: Int = 1,
        padding: Padding = .valid,
        activation: @escaping Activation = identity,
        filterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
        biasInitializer: ParameterInitializer<Scalar> = zeros()
    ) {
        let filterTensorShape = TensorShape([
            filterShape.0, filterShape.1, filterShape.2])
        self.init(
            filter: filterInitializer(filterTensorShape),
            bias: biasInitializer([filterShape.2]),
            activation: activation,
            stride: stride,
            padding: padding)
    }
}

/// A 2-D transposed convolution layer (e.g. spatial transposed convolution over images).
///
/// This layer creates a convolution filter that is transpose-convolved with the layer input
/// to produce a tensor of outputs.
@frozen
public struct TransposedConv2D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The 4-D convolution kernel.
    public var filter: Tensor<Scalar>
    /// The bias vector.
    public var bias: Tensor<Scalar>
    /// The element-wise activation function.
    @noDerivative public let activation: Activation
    /// The strides of the sliding window for spatial dimensions.
    @noDerivative public let strides: (Int, Int)
    /// The padding algorithm for convolution.
    @noDerivative public let padding: Padding
    /// The paddingIndex property allows us to handle computation based on padding.
    @noDerivative public let paddingIndex: Int

    /// The element-wise activation function type.
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

    /// Creates a `TransposedConv2D` layer with the specified filter, bias,
    /// activation function, strides, and padding.
    ///
    /// - Parameters:
    ///   - filter: A 4-D tensor of shape
    ///     `[height, width, output channel count, input channel count]`.
    ///   - bias: The bias tensor of shape `[output channel count]`.
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
        self.paddingIndex = padding == .same ? 0 : 1
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let batchSize = input.shape[0]
        let h = (input.shape[1] - (1 * paddingIndex)) *
          strides.0 + (filter.shape[0] * paddingIndex)
        let w = (input.shape[2] - (1 * paddingIndex)) *
          strides.1 + (filter.shape[1] * paddingIndex)
        let c = filter.shape[2]
        let newShape = Tensor<Int32>([Int32(batchSize), Int32(h), Int32(w), Int32(c)])
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
    /// element-wise activation function.
    ///
    /// - Parameters:
    ///   - filterShape: A 4-D tensor of shape
    ///     `[width, height, input channel count, output channel count]`.
    ///   - strides: The strides of the sliding window for spatial dimensions.
    ///   - padding: The padding algorithm for convolution.
    ///   - activation: The element-wise activation function.
    ///   - filterInitializer: Initializer to use for the filter parameters.
    ///   - biasInitializer: Initializer to use for the bias parameters.
    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        activation: @escaping Activation = identity,
        filterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
        biasInitializer: ParameterInitializer<Scalar> = zeros()
    ) {
        let filterTensorShape = TensorShape([
            filterShape.0, filterShape.1, filterShape.2, filterShape.3])
        self.init(
            filter: filterInitializer(filterTensorShape),
            bias: biasInitializer([filterShape.2]),
            activation: activation,
            strides: strides,
            padding: padding)
    }
}


/// A 3-D transposed convolution layer (e.g. spatial transposed convolution over images).
///
/// This layer creates a convolution filter that is transpose-convolved with the layer input
/// to produce a tensor of outputs.
@frozen
public struct TransposedConv3D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The 5-D convolution kernel.
    public var filter: Tensor<Scalar>
    /// The bias vector.
    public var bias: Tensor<Scalar>
    /// The element-wise activation function.
    @noDerivative public let activation: Activation
    /// The strides of the sliding window for spatial dimensions.
    @noDerivative public let strides: (Int, Int, Int)
    /// The padding algorithm for convolution.
    @noDerivative public let padding: Padding
    /// The paddingIndex property allows us to handle computation based on padding.
    @noDerivative public let paddingIndex: Int

    /// The element-wise activation function type.
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

    /// Creates a `TransposedConv3D` layer with the specified filter, bias,
    /// activation function, strides, and padding.
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
        self.paddingIndex = padding == .same ? 0 : 1
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let batchSize = input.shape[0]
        let w = (input.shape[1] - (1 * paddingIndex)) *
            strides.0 + (filter.shape[0] * paddingIndex)
        let h = (input.shape[2] - (1 * paddingIndex)) *
            strides.1 + (filter.shape[1] * paddingIndex)
        let d = (input.shape[3] - (1 * paddingIndex)) *
            strides.2 + (filter.shape[2] * paddingIndex)
        let c = filter.shape[3]
        let newShape = Tensor<Int32>([Int32(batchSize), Int32(w), Int32(h), Int32(d), Int32(c)])
        return activation(conv3DBackpropInput(
            input,
            shape: newShape,
            filter: filter,
            strides: (1, strides.0, strides.1, strides.2, 1),
            padding: padding) + bias)
    }
}

public extension TransposedConv3D {
    /// Creates a `TransposedConv3D` layer with the specified filter shape, strides, padding, and
    /// element-wise activation function. The filter tensor is initialized using Glorot uniform
    /// initialization with the specified generator. The bias vector is initialized with zeros.
    ///
    /// - Parameters:
    ///   - filterShape: The shape of the 5-D convolution kernel.
    ///   - strides: The strides of the sliding window for spatial dimensions.
    ///   - padding: The padding algorithm for convolution.
    ///   - activation: The element-wise activation function.
    ///   - generator: The random number generator for initialization.
    init(
        filterShape: (Int, Int, Int, Int, Int),
        strides: (Int, Int, Int) = (1, 1, 1),
        padding: Padding = .valid,
        activation: @escaping Activation = identity,
        filterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
        biasInitializer: ParameterInitializer<Scalar> = zeros()
    ) {
        let filterTensorShape = TensorShape([
            filterShape.0, filterShape.1, filterShape.2, filterShape.3, filterShape.4])
        self.init(
            filter: filterInitializer(filterTensorShape),
            bias: biasInitializer([filterShape.4]),
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
    /// The element-wise activation function.
    @noDerivative public let activation: Activation
    /// The strides of the sliding window for spatial dimensions.
    @noDerivative public let strides: (Int, Int)
    /// The padding algorithm for convolution.
    @noDerivative public let padding: Padding

    /// The element-wise activation function type.
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

    /// Creates a `DepthwiseConv2D` layer with the specified filter, bias, activation function,
    /// strides, and padding.
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
    /// - Parameter input: The input to the layer of shape,
    ///   [batch count, input height, input width, input channel count]
    /// - Returns: The output of shape,
    ///   [batch count, output height, output width, input channel count * channel multiplier]
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
    /// element-wise activation function.
    ///
    /// - Parameters:
    ///   - filterShape: The shape of the 4-D convolution kernel with form,
    ///     [filter width, filter height, input channel count, channel multiplier].
    ///   - strides: The strides of the sliding window for spatial/spatio-temporal dimensions.
    ///   - padding: The padding algorithm for convolution.
    ///   - activation: The element-wise activation function.
    ///   - filterInitializer: Initializer to use for the filter parameters.
    ///   - biasInitializer: Initializer to use for the bias parameters.
    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        activation: @escaping Activation = identity,
        filterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
        biasInitializer: ParameterInitializer<Scalar> = zeros()
    ) {
        let filterTensorShape = TensorShape([
            filterShape.0, filterShape.1, filterShape.2, filterShape.3])
        self.init(
            filter: filterInitializer(filterTensorShape),
            bias: biasInitializer([filterShape.2 * filterShape.3]),
            activation: activation,
            strides: strides,
            padding: padding)
    }
}

/// A layer for adding zero-padding in the temporal dimension.
public struct ZeroPadding1D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
    /// The padding values along the temporal dimension.
    @noDerivative public let padding: (Int, Int)

    /// Creates a zero-padding 1D Layer.
    ///
    /// - Parameter padding: A tuple of two integers describing how many zeros to be padded at the
    ///   beginning and end of the padding dimension.
    public init(padding: (Int, Int)) {
        self.padding = padding
    }

    /// Creates a zero-padding 1D Layer.
    ///
    /// - Parameter padding: An integer which describes how many zeros to be padded at the beginning
    ///   and end of the padding dimension.
    public init(padding: Int) {
        self.init(padding: (padding, padding))
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        input.padded(forSizes: [(0, 0), padding, (0, 0)])
    }
}

/// A layer for adding zero-padding in the spatial dimensions.
public struct ZeroPadding2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
    /// The padding values along the spatial dimensions.
    @noDerivative public let padding: ((Int, Int), (Int, Int))

    /// Creates a zero-padding 2D Layer.
    ///
    /// - Parameter padding: A tuple of 2 tuples of two integers describing how many zeros to
    ///   be padded at the beginning and end of each padding dimensions.
    public init(padding: ((Int, Int), (Int, Int))) {
        self.padding = padding
    }

    /// Creates a zero-padding 2D Layer.
    ///
    /// - Parameter padding: Tuple of 2 integers that describes how many zeros to be padded
    ///   at the beginning and end of each padding dimensions.
    public init(padding: (Int, Int)) {
        let (height, width) = padding
        self.init(padding: ((height, height), (width, width)))
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        input.padded(forSizes: [(0, 0), padding.0, padding.1, (0, 0)])
    }
}

/// A layer for adding zero-padding in the spatial/spatio-temporal dimensions.
public struct ZeroPadding3D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
    /// The padding values along the spatial/spatio-temporal dimensions.
    @noDerivative public let padding: ((Int, Int), (Int, Int), (Int, Int))

    /// Creates a zero-padding 3D Layer.
    ///
    /// - Parameter padding: A tuple of 3 tuples of two integers describing how many zeros to
    ///   be padded at the beginning and end of each padding dimensions.
    public init(padding: ((Int, Int), (Int, Int), (Int, Int))) {
        self.padding = padding
    }

    /// Creates a zero-padding 3D Layer.
    ///
    /// - Parameter padding: Tuple of 3 integers that describes how many zeros to be padded
    ///   at the beginning and end of each padding dimensions.
    public init(padding: (Int, Int, Int)) {
        let (height, width, depth) = padding
        self.init(padding: ((height, height), (width, width), (depth, depth)))
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        input.padded(forSizes: [(0, 0), padding.0, padding.1, padding.2, (0, 0)])
    }
}

/// A 1-D separable convolution layer.
///
/// This layer performs a depthwise convolution that acts separately on channels followed by
/// a pointwise convolution that mixes channels.
@frozen
public struct SeparableConv1D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The 3-D depthwise convolution kernel.
    public var depthwiseFilter: Tensor<Scalar>
    /// The 3-D pointwise convolution kernel.
    public var pointwiseFilter: Tensor<Scalar>
    /// The bias vector.
    public var bias: Tensor<Scalar>
    /// The element-wise activation function.
    @noDerivative public let activation: Activation
    /// The strides of the sliding window for spatial dimensions.
    @noDerivative public let stride: Int
    /// The padding algorithm for convolution.
    @noDerivative public let padding: Padding

    /// The element-wise activation function type.
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

    /// Creates a `SeparableConv1D` layer with the specified depthwise and pointwise filter,
    /// bias, activation function, strides, and padding.
    ///
    /// - Parameters:
    ///   - depthwiseFilter: The 3-D depthwise convolution kernel
    ///     `[filter width, input channels count, channel multiplier]`.
    ///   - pointwiseFilter: The 3-D pointwise convolution kernel
    ///     `[1, channel multiplier * input channels count, output channels count]`.
    ///   - bias: The bias vector.
    ///   - activation: The element-wise activation function.
    ///   - strides: The strides of the sliding window for spatial dimensions.
    ///   - padding: The padding algorithm for convolution.
    public init(
        depthwiseFilter: Tensor<Scalar>,
        pointwiseFilter: Tensor<Scalar>,
        bias: Tensor<Scalar>,
        activation: @escaping Activation = identity,
        stride: Int = 1,
        padding: Padding = .valid
    ) {
        self.depthwiseFilter = depthwiseFilter
        self.pointwiseFilter = pointwiseFilter
        self.bias = bias
        self.activation = activation
        self.stride = stride
        self.padding = padding
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let depthwise = depthwiseConv2D(
            input.expandingShape(at: 1),
            filter: depthwiseFilter.expandingShape(at: 1),
            strides: (1, stride, stride, 1),
            padding: padding)
        let x = conv2D(
            depthwise,
            filter: pointwiseFilter.expandingShape(at: 1),
            strides: (1, 1, 1, 1),
            padding: padding,
            dilations: (1, 1, 1, 1))
        return activation(x.squeezingShape(at: 1) + bias)
    }
}

public extension SeparableConv1D {
    /// Creates a `SeparableConv1D` layer with the specified depthwise and pointwise filter shape,
    /// strides, padding, and element-wise activation function.
    ///
    /// - Parameters:
    ///   - depthwiseFilterShape: The shape of the 3-D depthwise convolution kernel.
    ///   - pointwiseFilterShape: The shape of the 3-D pointwise convolution kernel.
    ///   - strides: The strides of the sliding window for temporal dimensions.
    ///   - padding: The padding algorithm for convolution.
    ///   - activation: The element-wise activation function.
    ///   - filterInitializer: Initializer to use for the filter parameters.
    ///   - biasInitializer: Initializer to use for the bias parameters.
    init(
        depthwiseFilterShape: (Int, Int, Int),
        pointwiseFilterShape: (Int, Int, Int),
        stride: Int = 1,
        padding: Padding = .valid,
        activation: @escaping Activation = identity,
        depthwiseFilterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
        pointwiseFilterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
        biasInitializer: ParameterInitializer<Scalar> = zeros()
    ) {
        let depthwiseFilterTensorShape = TensorShape([
            depthwiseFilterShape.0, depthwiseFilterShape.1, depthwiseFilterShape.2])
        let pointwiseFilterTensorShape = TensorShape([
            pointwiseFilterShape.0, pointwiseFilterShape.1, pointwiseFilterShape.2])
        self.init(
            depthwiseFilter: depthwiseFilterInitializer(depthwiseFilterTensorShape),
            pointwiseFilter: pointwiseFilterInitializer(pointwiseFilterTensorShape),
            bias: biasInitializer([pointwiseFilterShape.2]),
            activation: activation,
            stride: stride,
            padding: padding)
    }
}

/// A 2-D Separable convolution layer.
///
/// This layer performs a depthwise convolution that acts separately on channels followed by
/// a pointwise convolution that mixes channels.
@frozen
public struct SeparableConv2D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The 4-D depthwise convolution kernel.
    public var depthwiseFilter: Tensor<Scalar>
    /// The 4-D pointwise convolution kernel.
    public var pointwiseFilter: Tensor<Scalar>
    /// The bias vector.
    public var bias: Tensor<Scalar>
    /// The element-wise activation function.
    @noDerivative public let activation: Activation
    /// The strides of the sliding window for spatial dimensions.
    @noDerivative public let strides: (Int, Int)
    /// The padding algorithm for convolution.
    @noDerivative public let padding: Padding

    /// The element-wise activation function type.
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

    /// Creates a `SeparableConv2D` layer with the specified depthwise and pointwise filter,
    /// bias, activation function, strides, and padding.
    ///
    /// - Parameters:
    ///   - depthwiseFilter: The 4-D depthwise convolution kernel
    ///     `[filter height, filter width, input channels count, channel multiplier]`.
    ///   - pointwiseFilter: The 4-D pointwise convolution kernel
    ///     `[1, 1, channel multiplier * input channels count, output channels count]`.
    ///   - bias: The bias vector.
    ///   - activation: The element-wise activation function.
    ///   - strides: The strides of the sliding window for spatial dimensions.
    ///   - padding: The padding algorithm for convolution.
    public init(
        depthwiseFilter: Tensor<Scalar>,
        pointwiseFilter: Tensor<Scalar>,
        bias: Tensor<Scalar>,
        activation: @escaping Activation = identity,
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid
    ) {
        self.depthwiseFilter = depthwiseFilter
        self.pointwiseFilter = pointwiseFilter
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
        let depthwise = depthwiseConv2D(
            input,
            filter: depthwiseFilter,
            strides: (1, strides.0, strides.1, 1),
            padding: padding)
        return activation(conv2D(
            depthwise,
            filter: pointwiseFilter,
            strides: (1, 1, 1, 1),
            padding: padding,
            dilations: (1, 1, 1, 1)) + bias)
    }
}

public extension SeparableConv2D {
    /// Creates a `SeparableConv2D` layer with the specified depthwise and pointwise filter shape,
    /// strides, padding, and element-wise activation function.
    ///
    /// - Parameters:
    ///   - depthwiseFilterShape: The shape of the 4-D depthwise convolution kernel.
    ///   - pointwiseFilterShape: The shape of the 4-D pointwise convolution kernel.
    ///   - strides: The strides of the sliding window for spatial/spatio-temporal dimensions.
    ///   - padding: The padding algorithm for convolution.
    ///   - activation: The element-wise activation function.
    ///   - filterInitializer: Initializer to use for the filter parameters.
    ///   - biasInitializer: Initializer to use for the bias parameters.
    init(
        depthwiseFilterShape: (Int, Int, Int, Int),
        pointwiseFilterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        activation: @escaping Activation = identity,
        depthwiseFilterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
        pointwiseFilterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
        biasInitializer: ParameterInitializer<Scalar> = zeros()
    ) {
        let depthwiseFilterTensorShape = TensorShape([
            depthwiseFilterShape.0, depthwiseFilterShape.1, depthwiseFilterShape.2,
            depthwiseFilterShape.3])
        let pointwiseFilterTensorShape = TensorShape([
            pointwiseFilterShape.0, pointwiseFilterShape.1, pointwiseFilterShape.2,
            pointwiseFilterShape.3])
        self.init(
            depthwiseFilter: depthwiseFilterInitializer(depthwiseFilterTensorShape),
            pointwiseFilter: pointwiseFilterInitializer(pointwiseFilterTensorShape),
            bias: biasInitializer([pointwiseFilterShape.3]),
            activation: activation,
            strides: strides,
            padding: padding)
    }
}
