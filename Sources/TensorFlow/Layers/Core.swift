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
@frozen
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
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable(vjp: _vjpApplied(to:))
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        switch Context.local.learningPhase {
        case .training:
            return applyingTraining(to: input)
        case .inference:
            return applyingInference(to: input)
        }
    }

    @usableFromInline
    func _vjpApplied(to input: Tensor<Scalar>) ->
        (Tensor<Scalar>, (Tensor<Scalar>) ->
            (Dropout<Scalar>.TangentVector, Tensor<Scalar>)) {
        switch Context.local.learningPhase {
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

/// A flatten layer.
///
/// A flatten layer flattens the input when applied without affecting the batch size.
@frozen
public struct Flatten<Scalar: TensorFlowFloatingPoint>: Layer {
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
@frozen
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
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
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
        seed: (Int32, Int32) = (Int32.random(in: Int32.min..<Int32.max),
                                Int32.random(in: Int32.min..<Int32.max))
    ) {
        self.init(weight: Tensor(glorotUniform: [inputSize, outputSize],
                                 seed: seed),
                  bias: Tensor(zeros: [outputSize]),
                  activation: activation)
    }
}
