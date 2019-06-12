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
    /// The stride of the sliding window for temporal dimension.
    @noDerivative public let stride: Int
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
        self.stride = stride
        self.padding = padding
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer `[batchCount, width, inputChannels]`.
    /// - Returns: The output `[batchCount, newWidth, outputChannels]`.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let conv = conv2D(input.expandingShape(at: 1), filter: filter.expandingShape(at: 0),
                          strides: (1, 1, stride, 1), padding: padding)
        return activation(conv.squeezingShape(at: 1) + bias)
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
            filterShape.0, filterShape.1, filterShape.2])
        self.init(
            filter: Tensor(glorotUniform: filterTensorShape),
            bias: Tensor(zeros: TensorShape([filterShape.2])),
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
        seed: (Int32, Int32) = (Int32.random(in: Int32.min..<Int32.max),
                                Int32.random(in: Int32.min..<Int32.max))
    ) {
        let filterTensorShape = TensorShape([
            filterShape.0, filterShape.1, filterShape.2])
        self.init(
            filter: Tensor(glorotUniform: filterTensorShape, seed: seed),
            bias: Tensor(zeros: TensorShape([filterShape.2])),
            activation: activation,
            stride: stride,
            padding: padding)
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
        self.strides = strides
        self.padding = padding
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return activation(conv2D(input, filter: filter, strides: (1, strides.0, strides.1, 1),
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
            filterShape.0, filterShape.1, filterShape.2, filterShape.3])
        self.init(
            filter: Tensor(glorotUniform: filterTensorShape, generator: &generator),
            bias: Tensor(zeros: TensorShape([filterShape.3])),
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
        seed: (Int32, Int32) = (Int32.random(in: Int32.min..<Int32.max),
                                Int32.random(in: Int32.min..<Int32.max))
    ) {
        let filterTensorShape = TensorShape([
            filterShape.0, filterShape.1, filterShape.2, filterShape.3])
        self.init(
            filter: Tensor(glorotUniform: filterTensorShape, seed: seed),
            bias: Tensor(zeros: TensorShape([filterShape.3])),
            activation: activation,
            strides: strides,
            padding: padding)
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
        activation: @escaping Activation,
        strides: (Int, Int, Int),
        padding: Padding
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
        return activation(conv3D(input, filter: filter,
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
            bias: Tensor(zeros: TensorShape([filterShape.4])),
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
            bias: Tensor(zeros: TensorShape([filterShape.4])),
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
        return activation(conv2DBackpropInput(input, shape: newShape, filter: filter,
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
            filterShape.0, filterShape.1, filterShape.2, filterShape.3])
        self.init(
            filter: Tensor(glorotUniform: filterTensorShape, generator: &generator),
            bias: Tensor(zeros: TensorShape([filterShape.3])),
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
        seed: (Int32, Int32) = (Int32.random(in: Int32.min..<Int32.max),
                                Int32.random(in: Int32.min..<Int32.max))
    ) {
        let filterTensorShape = TensorShape([
            filterShape.0, filterShape.1, filterShape.2, filterShape.3])
        self.init(
            filter: Tensor(glorotUniform: filterTensorShape, seed: seed),
            bias: Tensor(zeros: TensorShape([filterShape.3])),
            activation: activation,
            strides: strides,
            padding: padding)
    }
}
