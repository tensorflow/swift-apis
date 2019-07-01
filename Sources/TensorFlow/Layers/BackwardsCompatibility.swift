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

// TODO: Remove this file after 0.4.

//===------------------------------------------------------------------------------------------===//
// Old Initialization Schemes
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
    init(
        inputSize: Int,
        outputSize: Int,
        activation: @escaping Activation = identity,
        seed: (Int32, Int32) = randomSeedForTensorFlow()
    ) {
        self.init(weight: Tensor(glorotUniform: [inputSize, outputSize],
                                 seed: seed),
                  bias: Tensor(zeros: [outputSize]),
                  activation: activation)
    }
}
