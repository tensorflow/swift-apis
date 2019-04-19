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
import TensorFlow
#endif

public extension Tensor {
    /// Creates a tensor from an array of tensors (which may themselves be scalars).
    @inlinable
    @differentiable(where Scalar : TensorFlowFloatingPoint)
    init(_ elements: [Tensor]) {
        self = Tensor(stacking: elements)
    }

    /// Stacks `tensors`, along the `axis` dimension, into a new tensor with rank one higher than
    /// the current tensor and each tensor in `tensors`.
    /// 
    /// Given that `tensors` all have shape `[A, B, C]`, and `tensors.count = N`, then:
    /// - if `axis == 0` then the resulting tensor will have the shape `[N, A, B, C]`.
    /// - if `axis == 1` then the resulting tensor will have the shape `[A, N, B, C]`.
    /// - etc.
    ///
    /// For example:
    /// ```
    /// // 'x' is [1, 4]
    /// // 'y' is [2, 5]
    /// // 'z' is [3, 6]
    /// Tensor(stacking: [x, y, z]) // is [[1, 4], [2, 5], [3, 6]]
    /// Tensor(stacking: [x, y, z], alongAxis: 1) // is [[1, 2, 3], [4, 5, 6]]
    /// ```
    ///
    /// This is the opposite of `Tensor.unstacked`.
    ///
    /// - Parameters:
    ///   - tensors: Tensors to stack.
    ///   - axis: Dimension along which to stack. Negative values wrap around.
    /// 
    /// - Precondition: All tensors must have the same shape.
    /// - Precondition: `axis` must be in the range `[-rank, rank)`, where `rank` is the rank of the
    ///   provided tensors.
    /// 
    /// - Returns: The stacked tensor.
    @inlinable
    @differentiable(vjp: _vjpStacking where Scalar : TensorFlowFloatingPoint)
    init(stacking tensors: [Tensor], alongAxis axis: Int = 0) {
        self = Raw.pack(tensors, axis: Int64(axis))
    }

    /// Concatenates `tensors` along the `axis` dimension.
    ///
    /// Given that `tensors[i].shape = [D0, D1, ... Daxis(i), ...Dn]`, then the concatenated result 
    /// has shape `[D0, D1, ... Raxis, ...Dn]`, where `Raxis = sum(Daxis(i))`. That is, the data 
    /// from the input tensors is joined along the `axis` dimension.
    ///
    /// For example:
    /// ```
    /// // t1 is [[1, 2, 3], [4, 5, 6]]
    /// // t2 is [[7, 8, 9], [10, 11, 12]]
    /// Tensor(concatenating: [t1, t2]) // is [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    /// Tensor(concatenating: [t1, t2], alongAxis: 1) // is [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
    /// 
    /// // t3 has shape [2, 3]
    /// // t4 has shape [2, 3]
    /// Tensor(concatenating: [t3, t4]) // has shape [4, 3]
    /// Tensor(concatenating: [t3, t4], alongAxis: 1) // has shape [2, 6]
    /// ```
    ///
    /// - Note: If you are concatenating along a new axis consider using 
    ///   `Tensor.init(stacking:alongAxis:)`.
    ///
    /// - Parameters:
    ///   - tensors: Tensors to concatenate.
    ///   - axis: Dimension along which to concatenate. Negative values wrap around.
    ///
    /// - Precondition: All tensors must have the same rank and all dimensions except `axis`
    ///   must be equal.
    /// - Precondition: `axis` must be in the range `[-rank, rank)`, where `rank` is the rank of the
    ///   provided tensors.
    /// 
    /// - Returns: The concatenated tensor.
    @inlinable
    @differentiable(wrt: tensors, vjp: _vjpConcatenating where Scalar : TensorFlowFloatingPoint)
    init(concatenating tensors: [Tensor], alongAxis axis: Int = 0) {
        precondition(tensors.count > 0)
        self = Raw.concatV2(tensors, axis: Tensor<Int32>(Int32(axis)))
    }

    /// Returns a tiled tensor, constructed by tiling the provided tensor.
    ///
    /// This constructor creates a new tensor by replicating `tensor` `multiples` times. The
    /// constructed tensor's `i`'th dimension has `tensor.shape[i] * multiples[i]` elements, and the
    /// values of `tensor` are replicated `multiples[i]` times along the `i`'th dimension. For 
    /// example, tiling `[a b c d]` by `[2]` produces `[a b c d a b c d]`.
    /// 
    /// - Precondition: The shape of `multiples` must be `[tensor.rank]`.
    @inlinable
    @differentiable(wrt: tensor, vjp: _vjpTiling where Scalar : TensorFlowFloatingPoint)
    init(tiling tensor: Tensor, multiples: Tensor<Int32>) {
        self = Raw.tile(tensor, multiples: multiples)
    }
}

internal extension Tensor where Scalar : TensorFlowFloatingPoint {
    @usableFromInline
    static func _vjpStacking(
        stacking tensors: [Tensor],
        alongAxis axis: Int = 0
    ) -> (Tensor, (Tensor) -> Array<Tensor>.DifferentiableView) {
        let result = Tensor(stacking: tensors, alongAxis: axis)
        return (result, { v in
            return Array<Tensor>.DifferentiableView(v.unstack(alongAxis: axis))
        })
    }

    @usableFromInline
    static func _vjpConcatenating(
        concatenating tensors: [Tensor],
        alongAxis axis: Int = 0
    ) -> (Tensor, (Tensor) -> Array<Tensor>.DifferentiableView) {
        let result = Tensor<Scalar>(concatenating: tensors, alongAxis: axis)
        let posAxis = axis < 0 ? axis + tensors[0].rank : axis
        let sizes = Tensor<Int32>(stacking: tensors.map { $0.shapeTensor[posAxis] })
        return (result, { [count = tensors.count] v in
            if count == 1 { return Array<Tensor>.DifferentiableView([v]) }
            let splits = v.split(sizes: sizes, alongAxis: posAxis)
            return Array<Tensor>.DifferentiableView(splits)
        })
    }

    @usableFromInline
    static func _vjpTiling(
        tiling tensor: Tensor<Scalar>,
        multiples: Tensor<Int32>
    ) -> (Tensor, (Tensor) -> Tensor) {
        let result = Tensor(tiling: tensor, multiples: multiples)
        return (result, { [shape = tensor.shapeTensor] v in
            let splitShape = Tensor<Int32>(stacking: [multiples, shape]).transposed().flattened()
            let axes = Tensor<Int32>(
                rangeFrom: 0, to: Int32(splitShape.scalarCount), stride: 2)
            return v.reshaped(toShape: splitShape).sum(squeezingAxes: axes)
        })
    }
}

public extension Tensor where Scalar : Numeric {
    /// Creates a tensor with all scalars set to zero that has the same shape and type as the provided 
    /// tensor.
    ///
    /// - Parameter other: Tensor whose shape and data type to use.
    @inlinable
    init(zerosLike other: Tensor) {
        self = Raw.zerosLike(other)
    }

    /// Creates a tensor with all scalars set to one that has the same shape and type as the provided 
    /// tensor.
    ///
    /// - Parameter other: Tensor whose shape and data type to use.
    @inlinable
    init(onesLike other: Tensor) {
        self = Raw.onesLike(other)
    }

    /// Creates a 1-D tensor representing a sequence from a starting value to, but not including, an 
    /// end value, stepping by the specified amount.
    ///
    /// - Parameters:
    ///   - start: The starting value to use for the sequence. If the sequence contains any values, 
    ///     the first one is `start`.
    ///   - end: An end value to limit the sequence. `end` is never an element of the resulting 
    ///     sequence.
    ///   - stride: The amount to step by with each iteration. `stride` must be positive.
    @inlinable
    init(rangeFrom start: Tensor<Scalar>, to end: Tensor<Scalar>, stride: Tensor<Scalar>) {
        self = Raw.range(start: start, limit: end, delta: stride)
    }
}

public extension Tensor where Scalar == Int32 {
    /// Creates a tensor with the specified shape, randomly sampling scalar values
    /// from a discrete uniform distribution.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///   - generator: Random number generator to use.
    ///
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
    ///
    init(randomStandardUniform shape: TensorShape) {
        self.init(randomStandardUniform: shape, generator: &PhiloxRandomNumberGenerator.global)
    }
}

public extension Tensor where Scalar: BinaryFloatingPoint {
    /// Creates a tensor with the specified shape, randomly sampling scalar values
    /// from a uniform distribution between 0 and 1, using the default random
    /// number generator.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///   - seed: The seed value.
    ///
    init(
        randomUniform shape: TensorShape,
        seed: (Int64, Int64) = (Int64.random(in: Int64.min..<Int64.max),
                                Int64.random(in: Int64.min..<Int64.max))
    ) {
        self = Raw.statelessRandomUniform(
          shape: Tensor<Int32>((0..<shape.rank).map { Int32(shape[$0]) }),
          seed: Tensor<Int64>([seed.0, seed.1])
        )
    }

    /// Creates a tensor with the specified shape, randomly sampling scalar values
    /// from a normal distribution, using the default random number generator.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///   - seed: The seed value.
    ///
    init(
        randomNormal shape: TensorShape,
        seed: (Int64, Int64) = (Int64.random(in: Int64.min..<Int64.max),
                                Int64.random(in: Int64.min..<Int64.max))
    ) {
        self = Raw.statelessRandomNormal(
            shape: Tensor<Int32>((0..<shape.rank).map { Int32(shape[$0]) }),
            seed: Tensor<Int64>([seed.0, seed.1])
        )
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
    init<G: RandomNumberGenerator>(
        randomUniform shape: TensorShape,
        generator: inout G
    ) {
        let dist = UniformFloatingPointDistribution<Scalar>()
        var scalars: [Scalar] = []
        for _ in 0 ..< shape.contiguousSize {
            scalars.append(dist.next(using: &generator))
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
    init<G: RandomNumberGenerator>(
        randomNormal shape: TensorShape,
        mean: Scalar = 0,
        stddev: Scalar = 1,
        generator: inout G
    ) {
        let dist = NormalDistribution<Scalar>(mean: mean, standardDeviation: stddev)
        var scalars: [Scalar] = []
        for _ in 0 ..< shape.contiguousSize {
            scalars.append(dist.next(using: &generator))
        }
        self.init(shape: shape, scalars: scalars)
    }
}

fileprivate extension Tensor where Scalar : BinaryFloatingPoint {
    private static func glorot(
        fromStandardUniform randomUniform: __shared Tensor<Scalar>,
        shape: __shared TensorShape
    ) -> Tensor<Scalar> {
        let spatialDimCount = shape.count - 2
        let receptiveField = shape[0..<spatialDimCount].contiguousSize
        let fanIn = shape[shape.count - 2] * receptiveField
        let fanOut = shape[shape.count - 1] * receptiveField
        let minusOneToOne = 2 * randomUniform - 1
        return sqrt(Scalar(6) / Scalar(fanIn + fanOut)) * minusOneToOne
    }
}

public extension Tensor where Scalar: TensorFlowFloatingPoint {
    /// Creates a tensor by performing Glorot uniform initialization for the specified shape,
    /// randomly sampling scalar values from a uniform distribution between `-limit` and `limit`,
    /// generated by the default random number generator, where limit is
    /// `sqrt(6 / (fanIn + fanOut))` and `fanIn`/`fanOut` represent the number of input and output
    /// features multiplied by the receptive field if present.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///
    init(
        glorotUniform shape: TensorShape,
        seed: (Int64, Int64) = (Int64.random(in: Int64.min..<Int64.max),
                                Int64.random(in: Int64.min..<Int64.max))
    ) {
        let uniform = Tensor(randomUniform: shape, seed: seed)
        self = Tensor.glorot(fromStandardUniform: uniform, shape: shape)
    }
}

public extension Tensor where Scalar: BinaryFloatingPoint,
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
        let uniform = Tensor(randomUniform: shape, generator: &generator)
        self = Tensor.glorot(fromStandardUniform: uniform, shape: shape)
    }
}
