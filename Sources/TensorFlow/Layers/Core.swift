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


/// Returns the result of dropout according to the current learning phase.
@differentiable(wrt: input)
private func dropout<Scalar: TensorFlowFloatingPoint>(
    _ input: Tensor<Scalar>,
    probability: Double,
    noiseShape: TensorShape
) -> Tensor<Scalar> {
    switch Context.local.learningPhase {
    case .training:
        return input.droppingOut(probability: probability, noiseShape: noiseShape)
    case .inference:
        return input
    }
}

/// A dropout layer.
///
/// For each update step of the training phase, `Dropout` randomly and independently omits input
/// units by setting them to `0` with probability, `probability`, according to the standard Uniform
/// distribution.
///
/// This layer effectively provides a way to train an ensemble of "thinned" out models of a
/// network in order to prevent the co-adaptation of units. By doing so, overfitting can be
/// significantly reduced.
///
/// Reference: ["Dropout: A Simple Way to Prevent Neural Networks from Overfitting"]
/// (http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
@frozen
public struct Dropout<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
    @noDerivative public let probability: Double

    /// Creates a dropout layer.
    ///
    /// - Parameter probability: The probability that an input unit will be dropped. This value must
    ///   be in the interval `0..<1`.
    public init(probability: Double) {
        precondition(
            (0..<1.0).contains(probability),
            "Dropout probability must be between 0 and 1. Got: \(probability)")
        self.probability = probability
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: An output of the same shape as `input`.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        dropout(input, probability: probability, noiseShape: input.shape)
    }
}

/// A 1-D spatial dropout layer.
///
/// For each update step of the training phase, `SpatialDropout1D` performs the same functionality
/// as `Dropout` but instead omits entire 1-D feature maps as opposed to individual units. This is
/// done to avoid possibly removing a single unit from a highly correlated neighborhood of units
/// (e.g. pixels), which results in greater independence.
///
/// Reference: [Efficient Object Localization Using Convolutional Networks]
/// (https://arxiv.org/pdf/1411.4280.pdf)
public struct SpatialDropout1D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
    @noDerivative public let probability: Double

    /// Creates a 1-D spatial dropout layer.
    ///
    /// - Parameter probability: The probability that an input unit will be dropped. This value must
    ///   be in the interval `0..<1`.
    public init(probability: Double) {
        precondition(
            (0..<1.0).contains(probability),
            "Dropout probability must be between 0 and 1. Got: \(probability)")
        self.probability = probability
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer of shape
    ///   [batch count, timestep count, channel count]
    /// - Returns: An output of the same shape as `input`.
    /// - Precondition: `input` must be rank `3`.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        dropout(input, probability: probability, noiseShape: [input.shape[0], 1, input.shape[2]])
    }
}

/// A 2-D spatial dropout layer.
///
/// For each update step of the training phase, `SpatialDropout2D` performs the same functionality
/// as `Dropout` but instead omits entire 2-D feature maps as opposed to individual units. This is
/// done to avoid possibly removing a single unit from a highly correlated neighborhood of units
/// (e.g. pixels), which results in greater independence.
///
/// Reference: [Efficient Object Localization Using Convolutional Networks]
/// (https://arxiv.org/pdf/1411.4280.pdf)
public struct SpatialDropout2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
    @noDerivative public let probability: Double

    /// Creates a 2-D spatial dropout layer.
    ///
    /// - Parameter probability: The probability that an input unit will be dropped. This value must
    ///   be in the interval `0..<1`.
    public init(probability: Double) {
        precondition(
            (0..<1.0).contains(probability),
            "Dropout probability must be between 0 and 1. Got: \(probability)")
        self.probability = probability
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer of shape
    ///   [batch count, height, width, channel count]
    /// - Returns: An output of the same shape as `input`.
    /// - Precondition: `input` must be rank `4`.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        dropout(input, probability: probability, noiseShape: [input.shape[0], 1, 1, input.shape[3]])
    }
}

/// A 3-D spatial dropout layer.
///
/// For each update step of the training phase, `SpatialDropout3D` performs the same functionality
/// as `Dropout` but instead omits entire 3-D feature maps as opposed to individual units. This is
/// done to avoid possibly removing a single unit from a highly correlated neighborhood of units
/// (e.g. pixels), which results in greater independence.
///
/// Reference: [Efficient Object Localization Using Convolutional Networks]
/// (https://arxiv.org/pdf/1411.4280.pdf)
public struct SpatialDropout3D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
    @noDerivative public let probability: Double

    /// Creates a 3-D spatial dropout layer.
    ///
    /// - Parameter probability: The probability that an input unit will be dropped. This value must
    ///   be in the interval `0..<1`.
    public init(probability: Double) {
        precondition(
            (0..<1.0).contains(probability),
            "Dropout probability must be between 0 and 1. Got: \(probability)")
        self.probability = probability
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer of shape
    ///   [batch count, depth, height, width, channel count]
    /// - Returns: An output of the same shape as `input`.
    /// - Precondition: `input` must be rank `5`.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        dropout(
            input,
            probability: probability,
            noiseShape: [input.shape[0], 1, 1, 1, input.shape[4]])
    }
}

/// A flatten layer.
///
/// A flatten layer flattens the input when applied without affecting the batch size.
@frozen
public struct Flatten<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
    /// Creates a flatten layer.
    public init() {}

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let batchSize = input.shape[0]
        let remaining = input.shape[1..<input.rank].contiguousSize
        return input.reshaped(to: [batchSize, remaining])
    }
}

/// A reshape layer.
@frozen
public struct Reshape<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
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
        self.init(shape: Tensor(shape.dimensions.map(Int32.init)))
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.reshaped(toShape: shape)
    }
}

/// A densely-connected neural network layer.
///
/// `Dense` implements the operation `activation(matmul(input, weight) + bias)`, where `weight` is
/// a weight matrix, `bias` is a bias vector, and `activation` is an element-wise activation
/// function.
///
/// This layer also supports 3-D weight tensors with 2-D bias matrices. In this case the first
/// dimension of both is treated as the batch size that is aligned with the first dimension of
/// `input` and the batch variant of the `matmul(_:_:)` operation is used, thus using a different
/// weight and bias for each element in input batch.
@frozen
public struct Dense<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The weight matrix.
    public var weight: Tensor<Scalar>
    /// The bias vector.
    public var bias: Tensor<Scalar>
    /// The element-wise activation function.
    @noDerivative public let activation: Activation
    /// Indicates whether this is a batched dense layer.
    @noDerivative internal let batched: Bool
    
    /// The element-wise activation function type.
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

    public init(
        weight: Tensor<Scalar>,
        bias: Tensor<Scalar>,
        activation: @escaping Activation
    ) {
        precondition(weight.rank <= 3, "The rank of the 'weight' tensor must be less than 4.")
        precondition(bias.rank <= 2, "The rank of the 'bias' tensor must be less than 3.")
        self.weight = weight
        self.bias = bias
        self.activation = activation
        self.batched = weight.rank == 3
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        if batched {
            let hidden = matmul(input.expandingShape(at: 1), weight)
            return activation(hidden.squeezingShape(at: 1) + bias)
        }
        return activation(matmul(input, weight) + bias)
    }
}

public extension Dense {
    /// Creates a `Dense` layer with the specified input size, output size, and element-wise
    /// activation function. The weight matrix is created with shape `[inputSize, outputSize]` and
    /// the bias vector is created with shape `[outputSize]`.
    ///
    /// - Parameters:
    ///   - inputSize: The dimensionality of the input space.
    ///   - outputSize: The dimensionality of the output space.
    ///   - activation: The activation function to use. The default value is `identity(_:)`.
    ///   - weightInitializer: Initializer to use for `weight`.
    ///   - biasInitializer: Initializer to use for `bias`.
    init(
        inputSize: Int,
        outputSize: Int,
        activation: @escaping Activation = identity,
        weightInitializer: ParameterInitializer<Scalar> = glorotUniform(),
        biasInitializer: ParameterInitializer<Scalar> = zeros()
    ) {
        self.init(
            weight: weightInitializer([inputSize, outputSize]),
            bias: biasInitializer([outputSize]),
            activation: activation)
    }
}

/// A layer that encloses a custom differentiable function.
public struct Function<Input: Differentiable, Output: Differentiable>: ParameterlessLayer {
    public typealias Body = @differentiable (Input) -> Output

    @noDerivative public let body: Body

    public init(_ body: @escaping Body) {
        self.body = body
    }

    @differentiable
    public func callAsFunction(_ input: Input) -> Output {
        body(input)
    }
}
