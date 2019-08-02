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
public struct Dropout<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
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
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        switch Context.local.learningPhase {
        case .training:
            return applyingTraining(to: input)
        case .inference:
            return applyingInference(to: input)
        }
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
            bias: Tensor(zeros: [outputSize]),
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
