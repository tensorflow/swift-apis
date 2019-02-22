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

public extension Tensor where Scalar == Int32 {
    /// Creates a tensor with the specified shape, randomly sampling scalar values
    /// from a discrete uniform distribution.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///   - generator: Random number generator to use.
    ///
    init<G: RandomNumberGenerator>(randomStandardUniform shape: TensorShape,
                                   generator: inout G) {
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
    ///
    init(randomStandardUniform shape: TensorShape) {
        let dist = UniformIntegerDistribution<Scalar>()
        var scalars: [Scalar] = []
        for _ in 0 ..< shape.contiguousSize {
            scalars.append(dist.next(using: &PhiloxRandomNumberGenerator.global))
        }
        self.init(shape: shape, scalars: scalars)
    }
}

public extension Tensor where Scalar: BinaryFloatingPoint,
                              Scalar.RawSignificand: FixedWidthInteger {
    /// Creates a tensor with the specified shape, randomly sampling scalar values
    /// from a uniform distribution between 0 and 1.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///   - generator: Random number generator to use.
    ///
    init<G: RandomNumberGenerator>(randomUniform shape: TensorShape,
                                   generator: inout G) {
        let dist = UniformFloatingPointDistribution<Scalar>()
        var scalars: [Scalar] = []
        for _ in 0 ..< shape.contiguousSize {
            scalars.append(dist.next(using: &generator))
        }
        self.init(shape: shape, scalars: scalars)
    }

    /// Creates a tensor with the specified shape, randomly sampling scalar values
    /// from a uniform distribution between 0 and 1, using the default random
    /// number generator.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///
    init(randomUniform shape: TensorShape) {
        let dist = UniformFloatingPointDistribution<Scalar>()
        var scalars: [Scalar] = []
        for _ in 0 ..< shape.contiguousSize {
            scalars.append(dist.next(using: &PhiloxRandomNumberGenerator.global))
        }
        self.init(shape: shape, scalars: scalars)
    }

    /// Creates a tensor with the specified shape, randomly sampling scalar values
    /// from a normal distribution.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///   - mean: The mean of the distribution.
    ///   - stddev: The standard deviation of the distribution.
    ///   - generator: Random number generator to use.
    ///
    init<G: RandomNumberGenerator>(randomNormal shape: TensorShape,
                                   mean: Scalar = 0,
                                   stddev: Scalar = 1,
                                   generator: inout G) {
        let dist = NormalDistribution<Scalar>(mean: mean, standardDeviation: stddev)
        var scalars: [Scalar] = []
        for _ in 0 ..< shape.contiguousSize {
            scalars.append(dist.next(using: &generator))
        }
        self.init(shape: shape, scalars: scalars)
    }

    /// Creates a tensor with the specified shape, randomly sampling scalar values
    /// from a normal distribution, using the default random number generator.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///   - mean: The mean of the distribution.
    ///   - stddev: The standard deviation of the distribution.
    ///
    init(randomNormal shape: TensorShape, mean: Scalar = 0, stddev: Scalar = 1) {
        let dist = NormalDistribution<Scalar>(mean: mean, standardDeviation: stddev)
        var scalars: [Scalar] = []
        for _ in 0 ..< shape.contiguousSize {
            scalars.append(dist.next(using:&PhiloxRandomNumberGenerator.global))
        }
        self.init(shape: shape, scalars: scalars)
    }
}

public extension Tensor where Scalar: TensorFlowFloatingPoint,
                              Scalar.RawSignificand: FixedWidthInteger {
    /// Performs Glorot uniform initialization for the specified shape, creating a tensor by
    /// randomly sampling scalar values from a uniform distribution between `-limit` and `limit`,
    /// where limit is `sqrt(6 / (fanIn + fanOut))` and `fanIn`/`fanOut` represent the number of
    /// input and output features multiplied by the receptive field if present.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///   - generator: Random number generator to use.
    ///
    init<G: RandomNumberGenerator>(glorotUniform shape: TensorShape, generator: inout G) {
        let spatialDimCount = shape.count - 2
        let receptiveField = shape[0..<spatialDimCount].contiguousSize
        let fanIn = shape[shape.count - 2] * receptiveField
        let fanOut = shape[shape.count - 1] * receptiveField
        let minusOneToOne = 2 * Tensor(randomUniform: shape, generator: &generator) - 1
        self = sqrt(Scalar(6) / Scalar(fanIn + fanOut)) * minusOneToOne
    }

    /// Creates a tensor by performing Glorot uniform initialization for the specified shape,
    /// randomly sampling scalar values from a uniform distribution between `-limit` and `limit`,
    /// generated by the default random number generator, where limit is
    /// `sqrt(6 / (fanIn + fanOut))` and `fanIn`/`fanOut` represent the number of input and output
    /// features multiplied by the receptive field if present.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///
    init(glorotUniform shape: TensorShape) {
        self.init(glorotUniform: shape, generator: &PhiloxRandomNumberGenerator.global)
    }
}
