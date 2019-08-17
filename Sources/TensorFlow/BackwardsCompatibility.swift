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

//===------------------------------------------------------------------------------------------===//
// Layers
//===------------------------------------------------------------------------------------------===//
// TODO: Remove this file after 0.4.

public extension Tensor where Scalar == Int32 {
    /// Creates a tensor with the specified shape, randomly sampling scalar values from a discrete
    /// uniform distribution.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///   - generator: Random number generator to use.
    ///   - lowerBound: The lower bound of the distribution.
    ///   - upperBound: The upper bound of the distribution.
    @available(*, deprecated, message: "This API will be removed after Swift for TensorFlow 0.4.")
    init<G: RandomNumberGenerator>(
        randomUniform shape: TensorShape,
        generator: inout G,
        lowerBound: Scalar = Scalar.min,
        upperBound: Scalar = Scalar.max
    ) {
        let dist = UniformIntegerDistribution<Scalar>(
            lowerBound: lowerBound,
            upperBound: upperBound)
        var scalars: [Scalar] = []
        for _ in 0 ..< shape.contiguousSize {
            scalars.append(dist.next(using: &generator))
        }
        self.init(shape: shape, scalars: scalars)
    }

    /// Creates a tensor with the specified shape, randomly sampling scalar values from a discrete
    /// uniform distribution, using the default random number generator.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///   - lowerBound: The lower bound of the distribution.
    ///   - upperBound: The upper bound of the distribution.
    @available(*, deprecated, message: "This API will be removed after Swift for TensorFlow 0.4.")
    init(
        randomUniform shape: TensorShape,
        lowerBound: Scalar = Scalar.min,
        upperBound: Scalar = Scalar.max
    ) {
        self.init(
            randomUniform: shape,
            generator: &Context.local.randomNumberGenerator,
            lowerBound: lowerBound,
            upperBound: upperBound)
    }
}

public extension Tensor where Scalar: BinaryFloatingPoint,
                              Scalar.RawSignificand: FixedWidthInteger {
    /// Creates a tensor with the specified shape, randomly sampling scalar values from a uniform
    /// distribution between `lowerBound` and `upperBound`.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///   - generator: Random number generator to use.
    ///   - lowerBound: The lower bound of the distribution.
    ///   - upperBound: The upper bound of the distribution.
    @available(*, deprecated, message: "This API will be removed after Swift for TensorFlow 0.4.")
    init<G: RandomNumberGenerator>(
        randomUniform shape: TensorShape,
        generator: inout G,
        lowerBound: Scalar = 0,
        upperBound: Scalar = 1
    ) {
        let dist = UniformFloatingPointDistribution<Scalar>(
            lowerBound: lowerBound,
            upperBound: upperBound)
        var scalars: [Scalar] = []
        for _ in 0 ..< shape.contiguousSize {
            scalars.append(dist.next(using: &generator))
        }
        let sample = Tensor(shape: shape, scalars: scalars)
        self = (upperBound - lowerBound) * sample + lowerBound
    }

    /// Creates a tensor with the specified shape, randomly sampling scalar values from a normal
    /// distribution.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///   - generator: Random number generator to use.
    ///   - mean: The mean of the distribution.
    ///   - standardDeviation: The standard deviation of the distribution.
    @available(*, deprecated, message: "This API will be removed after Swift for TensorFlow 0.4.")
    init<G: RandomNumberGenerator>(
        randomNormal shape: TensorShape,
        generator: inout G,
        mean: Scalar = 0,
        standardDeviation: Scalar = 1
    ) {
        let dist = NormalDistribution<Scalar>(mean: mean, standardDeviation: standardDeviation)
        var scalars: [Scalar] = []
        for _ in 0 ..< shape.contiguousSize {
            scalars.append(dist.next(using: &generator))
        }
        let sample = Tensor(shape: shape, scalars: scalars)
        self = standardDeviation * sample + mean
    }
}

public extension Tensor where Scalar: TensorFlowFloatingPoint {
    /// Performs Glorot uniform initialization for the specified shape, creating a tensor by
    /// randomly sampling scalar values from a uniform distribution between `-limit` and `limit`,
    /// where limit is `sqrt(6 / (fanIn + fanOut))` and `fanIn`/`fanOut` represent the number of
    /// input and output features multiplied by the receptive field if present.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///   - generator: Random number generator to use.
    @available(*, deprecated, message: "This API will be removed after Swift for TensorFlow 0.4.")
    init<G: RandomNumberGenerator>(glorotUniform shape: TensorShape, generator: inout G) {
        let uniform = Tensor(randomUniform: shape, generator: &generator)
        self = Tensor.glorot(fromStandardUniform: uniform, shape: shape)
    }

    /// Performs Glorot normal initialization for the specified shape, creating a tensor by
    /// randomly sampling scalar values from a uniform distribution between `-limit` and `limit`,
    /// where limit is `sqrt(2 / (fanIn + fanOut))` and `fanIn`/`fanOut` represent the number of
    /// input and output features multiplied by the receptive field if present.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///   - generator: Random number generator to use.
    @available(*, deprecated, message: "This API will be removed after Swift for TensorFlow 0.4.")
    init<G: RandomNumberGenerator>(glorotNormal shape: TensorShape, generator: inout G) {
        let normal = Tensor(randomNormal: shape, generator: &generator)
        self = Tensor.glorot(fromStandardNormal: normal, shape: shape)
    }
}

//===------------------------------------------------------------------------------------------===//
// Old Initialization Schemes
//===------------------------------------------------------------------------------------------===//

public extension Tensor where Scalar == Int32 {
    /// Creates a tensor with the specified shape, randomly sampling scalar values
    /// from a discrete uniform distribution.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///   - generator: Random number generator to use.
    @available(*, deprecated, message: "This API will be removed after Swift for TensorFlow 0.4.")
    init<G: RandomNumberGenerator>(
        randomStandardUniform shape: TensorShape,
        generator: inout G
    ) {
        let dist = UniformIntegerDistribution<Scalar>()
        var scalars: [Scalar] = []
        for _ in 0 ..< shape.contiguousSize {
            scalars.append(dist.next(using: &generator))
        }
        self.init(shape: shape, scalars: scalars)
    }

    /// Creates a tensor with the specified shape, randomly sampling scalar values
    /// from a discrete uniform distribution, using the default random number
    /// generator.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    @available(*, deprecated, message: "This API will be removed after Swift for TensorFlow 0.4.")
    init(randomStandardUniform shape: TensorShape) {
        self.init(randomStandardUniform: shape, generator: &PhiloxRandomNumberGenerator.global)
    }
}

//===------------------------------------------------------------------------------------------===//
// Old Layer Initialization Schemes
//===------------------------------------------------------------------------------------------===//

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
    @available(*, deprecated, message: """
        This API will be removed after Swift for TensorFlow 0.4, please consider using
        `init(filterShape:stride:padding:dilation:activation:filterIntializer:biasInitializer:)`
        instead.
        """)
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
    ///   - padding: The padding algorithm for convolution.
    ///   - dilation: The dilation factor for the temporal dimension.
    ///   - activation: The element-wise activation function.
    ///   - seed: The random seed for initialization. The default value is random.
    @available(*, deprecated, message: """
        This API will be removed after Swift for TensorFlow 0.4, please consider using
        `init(filterShape:stride:padding:dilation:activation:filterIntializer:biasInitializer:)`
        instead.
        """)
    init(
        filterShape: (Int, Int, Int),
        stride: Int = 1,
        padding: Padding = .valid,
        dilation: Int = 1,
        activation: @escaping Activation = identity,
        seed: TensorFlowSeed = Context.local.randomSeed
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
    @available(*, deprecated, message: """
        This API will be removed after Swift for TensorFlow 0.4, please consider using
        `init(filterShape:strides:padding:dilations:activation:filterIntializer:biasInitializer:)`
        instead.
        """)
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
    @available(*, deprecated, message: """
        This API will be removed after Swift for TensorFlow 0.4, please consider using
        `init(filterShape:strides:padding:dilations:activation:filterIntializer:biasInitializer:)`
        instead.
        """)
    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        dilations: (Int, Int) = (1, 1),
        activation: @escaping Activation = identity,
        seed: TensorFlowSeed = Context.local.randomSeed
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
    @available(*, deprecated, message: """
        This API will be removed after Swift for TensorFlow 0.4, please consider using
        `init(filterShape:strides:padding:activation:filterIntializer:biasInitializer:)` instead.
        """)
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
    @available(*, deprecated, message: """
        This API will be removed after Swift for TensorFlow 0.4, please consider using
        `init(filterShape:strides:padding:activation:filterIntializer:biasInitializer:)` instead.
        """)
    init(
        filterShape: (Int, Int, Int, Int, Int),
        strides: (Int, Int, Int) = (1, 1, 1),
        padding: Padding = .valid,
        activation: @escaping Activation = identity,
        seed: TensorFlowSeed = Context.local.randomSeed
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
    @available(*, deprecated, message: """
        This API will be removed after Swift for TensorFlow 0.4, please consider using
        `init(filterShape:strides:padding:activation:filterIntializer:biasInitializer:)` instead.
        """)
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
    @available(*, deprecated, message: """
        This API will be removed after Swift for TensorFlow 0.4, please consider using
        `init(filterShape:strides:padding:activation:filterIntializer:biasInitializer:)` instead.
        """)
    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        activation: @escaping Activation = identity,
        seed: TensorFlowSeed = Context.local.randomSeed
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
    @available(*, deprecated, message: """
        This API will be removed after Swift for TensorFlow 0.4, please consider using
        `init(filterShape:strides:padding:activation:filterIntializer:biasInitializer:)` instead.
        """)
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
    @available(*, deprecated, message: """
        This API will be removed after Swift for TensorFlow 0.4, please consider using
        `init(filterShape:strides:padding:activation:filterIntializer:biasInitializer:)` instead.
        """)
    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        activation: @escaping Activation = identity,
        seed: TensorFlowSeed = Context.local.randomSeed
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
    @available(*, deprecated, message: "This API will be removed after Swift for TensorFlow 0.4.")
    init<G: RandomNumberGenerator>(
        inputSize: Int,
        outputSize: Int,
        activation: @escaping Activation = identity,
        generator: inout G
    ) {
        self.init(weight: Tensor(glorotUniform: [inputSize, outputSize],
                                 generator: &generator),
                  bias: Tensor(zeros: [outputSize]),
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
    @available(*, deprecated, message: "This API will be removed after Swift for TensorFlow 0.4.")
    init(
        inputSize: Int,
        outputSize: Int,
        activation: @escaping Activation = identity,
        seed: TensorFlowSeed = Context.local.randomSeed
    ) {
        self.init(weight: Tensor(glorotUniform: [inputSize, outputSize],
                                 seed: seed),
                  bias: Tensor(zeros: [outputSize]),
                  activation: activation)
    }
}

//===------------------------------------------------------------------------------------------===//
// Losses
//===------------------------------------------------------------------------------------------===//

/// Returns the L1 loss between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func l1Loss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    l1Loss(predicted: predicted, expected: expected, reduction: { $0.sum() })
}

/// Returns the L2 loss between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func l2Loss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    l2Loss(predicted: predicted, expected: expected, reduction: { $0.sum() })
}

/// Returns the hinge loss between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func hingeLoss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    hingeLoss(predicted: predicted, expected: expected, reduction: _mean)
}

/// Returns the squared hinge loss between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func squaredHingeLoss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    squaredHingeLoss(predicted: predicted, expected: expected, reduction: _mean)
}

/// Returns the categorical hinge loss between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func categoricalHingeLoss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    categoricalHingeLoss(predicted: predicted, expected: expected, reduction: _mean)
}

/// Returns the logarithm of the hyperbolic cosine of the error between predictions and
/// expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func logCoshLoss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    logCoshLoss(predicted: predicted, expected: expected, reduction: _mean)
}

/// Returns the Poisson loss between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func poissonLoss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    poissonLoss(predicted: predicted, expected: expected, reduction: _mean)
}

/// Returns the Kullback-Leibler divergence (KL divergence) between between expectations and
/// predictions. Given two distributions `p` and `q`, KL divergence computes `p * log(p / q)`.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func kullbackLeiblerDivergence<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    kullbackLeiblerDivergence(predicted: predicted, expected: expected, reduction: { $0.sum() })
}

/// Returns the softmax cross entropy (categorical cross entropy) between logits and labels.
///
/// - Parameters:
///   - logits: One-hot encoded outputs from a neural network.
///   - labels: Indices (zero-indexed) of the correct outputs.
@differentiable(wrt: logits)
public func softmaxCrossEntropy<Scalar: TensorFlowFloatingPoint>(
    logits: Tensor<Scalar>,
    probabilities: Tensor<Scalar>
) -> Tensor<Scalar> {
    softmaxCrossEntropy(logits: logits, probabilities: probabilities, reduction: _mean)
}

/// Returns the sigmoid cross entropy (binary cross entropy) between logits and labels.
/// - Parameters:
///   - logits: The unscaled output of a neural network.
///   - labels: Integer values that correspond to the correct output.
@differentiable(wrt: logits)
public func sigmoidCrossEntropy<Scalar: TensorFlowFloatingPoint>(
    logits: Tensor<Scalar>,
    labels: Tensor<Scalar>
) -> Tensor<Scalar> {
    sigmoidCrossEntropy(logits: logits, labels:labels, reduction: _mean)
}
