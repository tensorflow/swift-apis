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

public extension Layer {
    func valueWithPullback(at input: Input)
        -> (output: Output,
            pullback: (Output.CotangentVector)
                -> (layerGradient: CotangentVector, inputGradient: Input.CotangentVector)) {
        let (out, pullback) = _valueWithPullback(at: self, input, in: Self.applied(to:))
        return (out, pullback)
    }
}

@_fixed_layout
public struct Dense<Scalar>: Layer
    where Scalar : FloatingPoint & Differentiable & TensorFlowScalar {
    public var weight: Tensor<Scalar>
    public var bias: Tensor<Scalar>

    @differentiable(wrt: (self, input))
    public func applied(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        return matmul(input, weight) + bias
    }
}

public extension Dense where Scalar : BinaryFloatingPoint,
                             Scalar.RawSignificand : FixedWidthInteger {
    init(inputSize: Int, outputSize: Int) {
        self.init(weight: Tensor(
            glorotUniform: [Int32(inputSize), Int32(outputSize)]),
            bias: Tensor(zeros: [Int32(outputSize)]))
    }
}

@_fixed_layout
public struct Conv2D<Scalar>: Layer
    where Scalar : FloatingPoint & Differentiable & TensorFlowScalar {
    public var filter: Tensor<Scalar>
    @noDerivative public let strides: (Int32, Int32)
    @noDerivative public let padding: Padding

    @differentiable(wrt: (self, input))
    public func applied(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.convolved2D(withFilter: filter,
                                 strides: (1, strides.0, strides.1, 1),
                                 padding: padding)
    }
}

public extension Conv2D where Scalar : BinaryFloatingPoint,
                              Scalar.RawSignificand : FixedWidthInteger {
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
            strides: (Int32(strides.0), Int32(strides.1)), padding: padding)
    }
}

@_fixed_layout
public struct BatchNorm<Scalar>: Layer
    where Scalar : BinaryFloatingPoint & Differentiable & TensorFlowScalar {
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
    @noDerivative public var runningMean: Tensor<Scalar>

    /// The running variance.
    @noDerivative public var runningVariance: Tensor<Scalar>

    @differentiable(wrt: (self, input))
    public func applied(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.batchNormalized(alongAxis: axis, offset: offset,
                                     scale: scale, epsilon: epsilon)
    }

    public init(axis: Int32,
                momentum: Tensor<Scalar> = Tensor(0.99),
                offset: Tensor<Scalar> = Tensor(0),
                scale: Tensor<Scalar> = Tensor(1),
                epsilon: Tensor<Scalar> = Tensor(0.001)) {
      self.axis = axis
      self.momentum = momentum
      self.offset = offset
      self.scale = scale
      self.epsilon = epsilon
      /// Initialize running mean and variance to zero.
      self.runningMean = Tensor(0)
      self.runningVariance = Tensor(1)
    }
}
