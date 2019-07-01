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
    /// - Parameter input: The input to the layer [batch count, input width, input channel count].
    /// - Returns: The output of shape [batch count, output width, output channel count],
    ///   where output width is computed as:
    ///
    ///   output width =
    ///   [(input width + 2 * padding size - (dilation * (filter width - 1) + 1)) / stride] + 1
    ///
    ///   and padding size is determined by the padding scheme. Note that padding size equals zero
    ///   when using .valid.
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
    ///     (filter width, input channel count, output channel count).
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
    ///     (filter width, input channel count, output channel count).
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
        seed: (Int32, Int32) = randomSeedForTensorFlow()
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
    /// The 4-D convolution filter.
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
    /// - Parameter input: The input to the layer of shape
    ///   [batch count, input height, input width, input channel count].
    /// - Returns: The output of shape
    ///   [batch count, output height, output width, output channel count],
    ///   where the output spatial dimensions are computed as:
    ///
    ///   output height =
    ///   [(input height + 2 * padding height - (dilation height * (filter height - 1) + 1))
    ///   / stride height] + 1
    ///
    ///   output width =
    ///   [(input width + 2 * padding width - (dilation width * (filter width - 1) + 1))
    ///   / stride width] + 1
    ///
    ///   and padding sizes are determined by the padding scheme. Note padding sizes equal zero
    ///   when using .valid.
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
    ///   - filterShape: The shape of the 4-D convolution filter, representing
    ///     (filter height, filter width, input channel count, output channel count).
    ///   - strides: The strides of the sliding window for spatial dimensions, i.e.
    ///     (stride height, stride width).
    ///   - padding: The padding algorithm for convolution.
    ///   - dilations: The dilation factors for spatial dimensions, i.e.
    ///     (dilation height, dilation width).
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
    ///   - filterShape: The shape of the 4-D convolution filter, representing
    ///     (filter height, filter width, input channel count, output channel count).
    ///   - strides: The strides of the sliding window for spatial dimensions, i.e.
    ///     (stride height, stride width).
    ///   - padding: The padding algorithm for convolution.
    ///   - dilations: The dilation factors for spatial dimensions, i.e.
    ///     (dilation height, dilation width).
    ///   - activation: The element-wise activation function.
    ///   - seed: The random seed for initialization. The default value is random.
    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        dilations: (Int, Int) = (1, 1),
        activation: @escaping Activation = identity,
        seed: (Int32, Int32) = randomSeedForTensorFlow()
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
    /// The 5-D convolution filter.
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
    ///   - filter: The 5-D convolution filter of shape
    ///     [filter depth, filter height, filter width, input channel count,
    ///     output channel count].
    ///   - bias: The bias vector of shape [output channel count].
    ///   - activation: The element-wise activation function.
    ///   - strides: The strides of the sliding window for spatial dimensions, i.e.
    ///     (stride depth, stride height, stride width)
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
    /// - Parameter input: The input to the layer of shape
    ///   [batch count, input depth, input height, input width, input channel count]
    /// - Returns: The output of shape
    ///   [batch count, output depth, output height, output width, output channel count]
    ///   where the spatial dimensions are computed as:
    ///
    ///   output depth =
    ///   [(input depth + 2 * padding depth - (dilation depth * (filter depth - 1) + 1))
    ///   / stride depth] + 1
    ///
    ///   output height =
    ///   [(input height + 2 * padding height - (dilation height * (filter height - 1) + 1))
    ///   / stride height] + 1
    ///
    ///   output width =
    ///   [(input width + 2 * padding width - (dilation width * (filter width - 1) + 1))
    ///   / stride width] + 1
    ///
    ///   and padding sizes are determined by the padding scheme. Note that padding sizes equal zero
    ///   when using .valid.
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
    ///   - filterShape: The shape of the 5-D convolution filter, representing
    ///     (filter depth, filter height, filter width, input channel count,
    ///     output channel count).
    ///   - strides: The strides of the sliding window for spatial dimensions, i.e.
    ///     (stride depth, stride height, stride width)
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
    ///   - filterShape: The shape of the 5-D convolution filter, representing
    ///     (filter depth, filter height, filter width, input channel count,
    ///     output channel count).
    ///   - strides: The strides of the sliding window for spatial dimensions, i.e.
    ///     (stride depth, stride height, stride width)
    ///   - padding: The padding algorithm for convolution.
    ///   - activation: The element-wise activation function.
    ///   - seed: The random seed for initialization. The default value is random.
    init(
        filterShape: (Int, Int, Int, Int, Int),
        strides: (Int, Int, Int) = (1, 1, 1),
        padding: Padding = .valid,
        activation: @escaping Activation = identity,
        seed: (Int32, Int32) = randomSeedForTensorFlow()
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
public struct TransposedConv2D<Scalar: TensorFlowFloatingPoint>: Layer {
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
        seed: (Int32, Int32) = randomSeedForTensorFlow()
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
        seed: (Int32, Int32) = randomSeedForTensorFlow()
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

/// A layer for adding zero-padding in the temporal dimension.
public struct ZeroPadding1D<Scalar: TensorFlowFloatingPoint>: Layer {
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
        return input.padded(forSizes: [(padding.0, padding.1)])
    }
}

/// A layer for adding zero-padding in the spatial dimensions.
public struct ZeroPadding2D<Scalar: TensorFlowFloatingPoint>: Layer {
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
        return input.padded(forSizes: [padding.0, padding.1])
    }
}

/// A layer for adding zero-padding in the spatial/spatio-temporal dimensions.
public struct ZeroPadding3D<Scalar: TensorFlowFloatingPoint>: Layer {
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
        return input.padded(forSizes: [padding.0, padding.1, padding.2])
    }
}
