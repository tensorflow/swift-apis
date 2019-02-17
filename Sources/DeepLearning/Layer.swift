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

/// A neural network layer.
///
/// Types that conform to `Layer` represent functions that map inputs to
/// outputs. They may have an internal state represented by parameters, such as
/// weight tensors.
///
/// `Layer` instances define a differentiable `applied(to:)` method for mapping
/// inputs to outputs.
public protocol Layer: Differentiable & KeyPathIterable
    where AllDifferentiableVariables: KeyPathIterable {
    /// The input type of the layer.
    associatedtype Input: Differentiable
    /// The output type of the layer.
    associatedtype Output: Differentiable

    /// Returns the output obtained from applying to an input.
    @differentiable(wrt: (self, input))
    func applied(to input: Input) -> Output
}

// FIXME: Extra where clause is a workaround for: SR-9595
public extension Layer
    where TangentVector : AdditiveArithmetic, CotangentVector : AdditiveArithmetic,
          Input.TangentVector : AdditiveArithmetic, Input.CotangentVector : AdditiveArithmetic {
    func valueWithPullback(at input: Input)
        -> (output: Output,
            pullback: (Output.CotangentVector)
                -> (layerGradient: CotangentVector, inputGradient: Input.CotangentVector)) {
        let (out, pullback) = valueWithPullback(at: input) { layer, input in
            return layer.applied(to: input)
        }
        return (out, pullback)
    }
}

/// A sequencing of two Layers to form a chain.
///
/// The sequential combinator composes two layers sequentially, feeding the
/// output of the first layer as input to the second layer.
///
/// It is normally constructed via the >> operator.
public struct Sequential<LHS: Layer, RHS: Layer>: Layer
    where LHS.Output == RHS.Input,
          LHS.TangentVector: AdditiveArithmetic,
          RHS.TangentVector: AdditiveArithmetic,
          LHS.CotangentVector: AdditiveArithmetic,
          RHS.CotangentVector: AdditiveArithmetic,
          LHS.Input.CotangentVector: AdditiveArithmetic,
          LHS.Output.CotangentVector: AdditiveArithmetic,
          RHS.Output.CotangentVector: AdditiveArithmetic,
          RHS.Output.TangentVector: AdditiveArithmetic {
    let lhs: LHS
    let rhs: RHS

     init(_ lhs: LHS, _ rhs: RHS) {
        self.lhs = lhs
        self.rhs = rhs
    }

    @differentiable(wrt: (self, input))
    public func applied(to input: LHS.Input) -> RHS.Output {
        let intermediateValue = lhs.applied(to: input)
        return rhs.applied(to: intermediateValue)
    }
}

public func >> <LHS: Layer, RHS: Layer>(_ lhs: LHS, _ rhs: RHS) -> Sequential<LHS, RHS> {
    return Sequential(lhs, rhs)
}

// TODO(TF-244): Uncomment once TF-244 is resolved.
//public struct Parallel<LHS: Layer, RHS: Layer, LHSOutput, RHSOutput, AggregateOutput: Differentiable>: Layer
//    where LHS.Output == LHSOutput,
//          RHS.Output == RHSOutput,
//          LHS.Input == RHS.Input,
//          LHS.TangentVector: AdditiveArithmetic,
//          RHS.TangentVector: AdditiveArithmetic,
//          LHS.CotangentVector: AdditiveArithmetic,
//          RHS.CotangentVector: AdditiveArithmetic,
//          LHS.Input.CotangentVector: AdditiveArithmetic,
//          LHS.Output.CotangentVector: AdditiveArithmetic,
//          RHS.Output.CotangentVector: AdditiveArithmetic,
//          RHS.Output.TangentVector: AdditiveArithmetic {
//    let lhs: LHS
//    let rhs: RHS
//    @noDerivative
//    let combiner: @differentiable (LHSOutput, RHSOutput) -> AggregateOutput
//
//     init(_ lhs: LHS, _ rhs: RHS, combinedWith combiner: @escaping @differentiable (LHSOutput, RHSOutput) -> AggregateOutput) {
//        self.lhs = lhs
//        self.rhs = rhs
//        self.combiner = combiner
//    }
//
//    @differentiable(wrt: (self, input))
//    public func applied(to input: LHS.Input) -> AggregateOutput {
//        return combiner(lhs.applied(to: input), rhs.applied(to: input))
//    }
//}

/// A mutable, shareable flag that denotes training vs. inference.
///
/// In typical uses, every layer in a model that has behavior which differs
/// between training and inference shares an instance of ModeRef so it doesn't
/// need to be toggled or threaded through in more than one place.
public final class LearningPhaseIndicator {
    public var training: Bool = true
    public init() {}
}

/// A mutable, shareable reference to a tensor
public final class Parameter<T: TensorFlowScalar> {
    public var value: Tensor<T>
    public init(_ value: Tensor<T>) {
        self.value = value
    }
}

@_fixed_layout
public struct Dense<Scalar: TensorFlowFloatingPoint>: Layer {

    public var weight: Tensor<Scalar>
    public var bias: Tensor<Scalar>
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
    @noDerivative public let activation: Activation

    // FIXME(SR-9716): Remove this once the bug is fixed or worked around.
    public var allKeyPaths: [PartialKeyPath<Dense>] {
        return [\Dense.weight, \Dense.bias]
    }

    @differentiable(wrt: (self, input))
    public func applied(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        return activation(matmul(input, weight) + bias)
    }
}

public extension Dense where Scalar.RawSignificand: FixedWidthInteger {
    init(inputSize: Int, outputSize: Int, activation: @escaping Activation) {
        self.init(weight: Tensor(
                  glorotUniform: [Int32(inputSize), Int32(outputSize)]),
                  bias: Tensor(zeros: [Int32(outputSize)]),
                  activation: activation)
    }
}

@_fixed_layout
public struct Conv2D<Scalar: TensorFlowFloatingPoint>: Layer {
    public var filter: Tensor<Scalar>
    public var bias: Tensor<Scalar>
    @noDerivative public let strides: (Int32, Int32)
    @noDerivative public let padding: Padding

    @differentiable(wrt: (self, input))
    public func applied(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.convolved2D(withFilter: filter,
                                 strides: (1, strides.0, strides.1, 1),
                                 padding: padding) + bias
    }
}

public extension Conv2D where Scalar.RawSignificand: FixedWidthInteger {
    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding
    ) {
        let filterTensorShape = TensorShape([
            Int32(filterShape.0), Int32(filterShape.1),
            Int32(filterShape.2), Int32(filterShape.3)])
        self.init(
            filter: Tensor(glorotUniform: filterTensorShape),
            bias: Tensor(zeros: TensorShape([Int32(filterShape.3)])),
            strides: (Int32(strides.0), Int32(strides.1)), padding: padding)
    }
}

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

    @noDerivative public let learningPhaseIndicator: LearningPhaseIndicator

    @differentiable(wrt: (self, input))
    private func applyTraining(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        let positiveAxis = (input.rank + axis) % input.rank
        let mean = input.mean(alongAxes: [0, positiveAxis])
        let variance = input.variance(alongAxes: [0, positiveAxis])
        runningMean.value += (mean - runningMean.value) * (1 - momentum)
        runningVariance.value += (
            variance - runningVariance.value) * (1 - momentum)
        let inv = rsqrt(variance + epsilon) * scale
        return (input - mean) * inv + offset
    }

    @differentiable(wrt: (self, input))
    private func applyInference(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        let inv = rsqrt(runningVariance.value + epsilon) * scale
        return (input - runningMean.value) * inv + offset
    }

    @differentiable(wrt: (self, input), vjp: _vjpApplied(to:))
    public func applied(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        if learningPhaseIndicator.training {
            return applyTraining(to: input)
        } else {
            return applyInference(to: input)
        }
    }

    @usableFromInline
    func _vjpApplied(to input: Tensor<Scalar>) ->
        (Tensor<Scalar>, (Tensor<Scalar>) ->
            (BatchNorm<Scalar>.CotangentVector, Tensor<Scalar>)) {
        if learningPhaseIndicator.training {
            return self.valueWithPullback(at: input) {
                $0.applyTraining(to: $1)
            }
        } else {
            return self.valueWithPullback(at: input) {
                $0.applyInference(to: $1)
            }
        }
    }

    public init(featureCount: Int,
                learningPhaseIndicator: LearningPhaseIndicator,
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
        self.learningPhaseIndicator = learningPhaseIndicator
    }
}

@_fixed_layout
public struct MaxPool2D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The size of the sliding reduction window for pooling.
    @noDerivative let poolSize: (Int32, Int32, Int32, Int32)

    /// The strides of the sliding window for each dimension of a 4-D input.
    /// Strides in non-spatial dimensions must be 1.
    @noDerivative let strides: (Int32, Int32, Int32, Int32)

    /// The padding algorithm for pooling.
    @noDerivative let padding: Padding

    // strides are just for the spatial dimensions (H and W)
    public init(poolSize: (Int, Int), strides: (Int, Int), padding: Padding) {
        self.poolSize = (1, Int32(poolSize.0), Int32(poolSize.1), 1)
        self.strides = (1, Int32(strides.0), Int32(strides.1), 1)
        self.padding = padding
    }

    @differentiable(wrt: (self, input))
    public func applied(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.maxPooled(
          kernelSize: poolSize, strides: strides, padding: padding)
    }
}

@_fixed_layout
public struct AvgPool2D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The size of the sliding reduction window for pooling.
    @noDerivative let poolSize: (Int32, Int32, Int32, Int32)

    /// The strides of the sliding window for each dimension of a 4-D input.
    /// Strides in non-spatial dimensions must be 1.
    @noDerivative let strides: (Int32, Int32, Int32, Int32)

    /// The padding algorithm for pooling.
    @noDerivative let padding: Padding

    // strides are just for the spatial dimensions (H and W)
    public init(poolSize: (Int, Int), strides: (Int, Int), padding: Padding) {
        self.poolSize = (1, Int32(poolSize.0), Int32(poolSize.1), 1)
        self.strides = (1, Int32(strides.0), Int32(strides.1), 1)
        self.padding = padding
    }

    @differentiable(wrt: (self, input))
    public func applied(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.averagePooled(
          kernelSize: poolSize, strides: strides, padding: padding)
    }
}

@_fixed_layout
public struct LayerNorm<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The offset value, also known as beta.
    public var offset: Tensor<Scalar>

    /// The scale value, also known as gamma.
    public var scale: Tensor<Scalar>

    @noDerivative public let axis: Int32

    /// The variance epsilon value.
    @noDerivative public let epsilon: Tensor<Scalar>

    public init(featureCount: Int,
                axis: Int,
                epsilon: Tensor<Scalar> = Tensor(0.001)) {
        self.scale = Tensor<Scalar>(ones: [Int32(featureCount)])
        self.offset = Tensor<Scalar>(zeros: [Int32(featureCount)])
        self.axis = Int32(axis)
        self.epsilon = epsilon
    }

    @differentiable(wrt: (self, input))
    public func applied(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        let mean = input.mean(alongAxes: axis)
        let variance = input.variance(alongAxes: axis)
        let inv = rsqrt(variance + epsilon) * scale
        return (input - mean) * inv + offset
    }
}

public extension Tensor
    where Scalar : TensorFlowFloatingPoint, Scalar.RawSignificand: FixedWidthInteger {
    @differentiable(wrt: self where Scalar: Differentiable)
    func droppingOut(probability: Double) -> Tensor {
        let noise = Tensor(randomUniform: shape)
        let keepMask = noise .>= Scalar(probability)
        let keepProbability = Scalar(1.0 - probability)
        return self * Tensor(keepMask) / Tensor(keepProbability)
    }
}

@_fixed_layout
public struct Dropout<Scalar: TensorFlowFloatingPoint>: Layer
    where Scalar.RawSignificand: FixedWidthInteger {
    @noDerivative public let probability: Double
    @noDerivative public let learningPhaseIndicator: LearningPhaseIndicator
    // Workaround for TF-189, making `Dropout` have a non-trivial parameter
    // convention.
    var _unused: Tensor<Scalar>

    public init(
        probability: Double,
        learningPhaseIndicator: LearningPhaseIndicator
    ) {
        self.probability = probability
        self.learningPhaseIndicator = learningPhaseIndicator
        // Workaround for TF-8
        self._unused = Tensor<Scalar>(0)
    }

    @differentiable(wrt: (self, input))
    private func applyTraining(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.droppingOut(probability: probability)
    }

    @differentiable(wrt: (self, input))
    private func applyInference(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input
    }

    @differentiable(wrt: (self, input), vjp: _vjpApplied(to:))
    public func applied(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        if learningPhaseIndicator.training {
            return applyTraining(to: input)
        } else {
            return applyInference(to: input)
        }
    }

    @usableFromInline
    func _vjpApplied(to input: Tensor<Scalar>) ->
        (Tensor<Scalar>, (Tensor<Scalar>) ->
            (Dropout<Scalar>.CotangentVector, Tensor<Scalar>)) {
        if learningPhaseIndicator.training {
            return self.valueWithPullback(at: input) {
                $0.applyTraining(to: $1)
            }
        } else {
            return self.valueWithPullback(at: input) {
                $0.applyInference(to: $1)
            }
        }
    }
}

@_fixed_layout
public struct UpSampling2D<Scalar: TensorFlowFloatingPoint>: Layer {
    @noDerivative public let size: Int32

    // FIXME(TF-25): Uncomment when the derived conformances bug is fixed.
    // public init(size: Int32) {
    //    self.size = size
    // }

    @differentiable(wrt: (self, input))
    public func applied(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        let batchSize = input.shape[0]
        let height = input.shape[1]
        let width = input.shape[2]
        let channels = input.shape[3]
        let reshapeSize = Tensor<Int32>([batchSize, height, 1, width, 1, channels])
        let scaleOnes = Tensor<Scalar>(ones: [1, 1, size, 1, size, 1])
        let upSampling = input.reshaped(toShape: reshapeSize) * scaleOnes
        let upSampledShape = Tensor<Int32>([batchSize, height * size, width * size, channels])
        let upSampled = upSampling.reshaped(toShape: upSampledShape)
        return upSampled
    }
}
