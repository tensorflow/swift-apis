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

import _Differentiation

#if !COMPILING_TENSORFLOW_STDLIB_MODULE
  import Tensor
#endif

extension Tensor {
  /// Creates a tensor with the specified shape and a single, repeated scalar value.
  ///
  /// - Parameters:
  ///   - shape: The dimensions of the tensor.
  ///   - repeatedValue: The scalar value to repeat.
  @inlinable
  @available(*, deprecated, renamed: "init(repeating:shape:)")
  public init(shape: TensorShape, repeating repeatedValue: Scalar) {
    self.init(repeating: repeatedValue, shape: shape)
  }

  /// Creates a tensor with the specified shape and a single, repeated scalar value.
  ///
  /// - Parameters:
  ///   - repeatedValue: The scalar value to repeat.
  ///   - shape: The dimensions of the tensor.
  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public init(
    repeating repeatedValue: Scalar, shape: TensorShape,
    on device: Device = .default
  ) {
    self = _Raw.fill(
      dims: Tensor<Int32>(shape.dimensions.map(Int32.init), on: device),
      value: Tensor(repeatedValue, on: device))
  }

  /// Creates a tensor by broadcasting the given scalar to a given rank with
  /// all dimensions being 1.
  @inlinable
  // @differentiable(where Scalar: TensorFlowFloatingPoint)
  public init(broadcasting scalar: Scalar, rank: Int, on device: Device = .default) {
    self = Tensor(scalar, on: device).reshaped(to: TensorShape(repeating: 1, count: rank))
  }

  /// Creates a tensor of shape `[4]` from a 4-tuple.
  /// - Note: This is intended for internal use, for example, to initialize a
  ///   tensor attribute from `convolved2D`'s `strides` argument.
  @inlinable
  internal init(_ scalars: (Scalar, Scalar, Scalar, Scalar), on device: Device = .default) {
    self.init([scalars.0, scalars.1, scalars.2, scalars.3], on: device)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: init(repeating:shape:on:))
  static func _vjpInit(
    repeating repeatedValue: __owned Scalar,
    shape: __owned TensorShape,
    on device: Device
  ) -> (value: Tensor, pullback: (Tensor) -> Scalar) {
    return (
      Tensor(repeating: repeatedValue, shape: shape, on: device),
      {
        $0.sum().scalarized()
      }
    )
  }
}

//===------------------------------------------------------------------------------------------===//
// Casting
//===------------------------------------------------------------------------------------------===//

extension Tensor where Scalar: Numeric {
  /// Perform an element-wise type conversion from a `Bool` tensor.
  @inlinable
  public init(_ other: Tensor<Bool>) {
    self = _Raw.cast(other)
  }

  /// Perform an element-wise conversion from another `Tensor`.
  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint, OtherScalar: TensorFlowFloatingPoint)
  public init<OtherScalar: Numeric>(_ other: Tensor<OtherScalar>) {
    self = _Raw.cast(other)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: init(_:))
  static func _vjpCast<OtherScalar: TensorFlowFloatingPoint>(
    _ other: __owned Tensor<OtherScalar>
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor<OtherScalar>) {
    (Tensor(other), { v in Tensor<OtherScalar>(v) })
  }
}

//===------------------------------------------------------------------------------------------===//
// Stacking / Concatenating / Tiling
//===------------------------------------------------------------------------------------------===//

extension Tensor {
  /// Creates a tensor from an array of tensors (which may themselves be scalars).
  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public init(_ elements: [Tensor]) {
    self = _Raw.pack(elements)
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
  /// This is the opposite of `Tensor.unstacked(alongAxis:)`.
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
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public init(stacking tensors: [Tensor], alongAxis axis: Int = 0) {
    self = _Raw.pack(tensors, axis: Int64(axis))
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
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public init(concatenating tensors: [Tensor], alongAxis axis: Int = 0) {
    precondition(tensors.count > 0)
    self = _Raw.concatV2(tensors, axis: Tensor<Int32>(Int32(axis), on: tensors.first!.device))
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: init(_:))
  static func _vjpInitElements(
    _ elements: __owned [Tensor]
  ) -> (value: Tensor, pullback: (Tensor) -> Array<Tensor>.DifferentiableView) {
    _vjpStacking(stacking: elements)
  }

  @inlinable
  @derivative(of: init(stacking:alongAxis:))
  static func _vjpStacking(
    stacking tensors: __owned [Tensor],
    alongAxis axis: __owned Int = 0
  ) -> (value: Tensor, pullback: (Tensor) -> Array<Tensor>.DifferentiableView) {
    (
      Tensor(stacking: tensors, alongAxis: axis),
      { v in
        Array<Tensor>.DifferentiableView(v.unstacked(alongAxis: axis))
      }
    )
  }

  @inlinable
  @derivative(of: init(concatenating:alongAxis:))
  static func _vjpConcatenating(
    concatenating tensors: __owned [Tensor],
    alongAxis axis: __owned Int = 0
  ) -> (value: Tensor, pullback: (Tensor) -> Array<Tensor>.DifferentiableView) {
    let result = Tensor<Scalar>(concatenating: tensors, alongAxis: axis)
    let posAxis = axis < 0 ? axis + tensors[0].rank : axis
    let sizes = Tensor<Int32>(stacking: tensors.map { $0.shapeTensor[posAxis] })
    return (
      result,
      { [count = tensors.count] v in
        if count == 1 { return Array<Tensor>.DifferentiableView([v]) }
        let splits = v.split(sizes: sizes, alongAxis: posAxis)
        return Array<Tensor>.DifferentiableView(splits)
      }
    )
  }
}

//===------------------------------------------------------------------------------------------===//
// Numeric
//===------------------------------------------------------------------------------------------===//

extension Tensor where Scalar: Numeric {
  /// Creates a tensor with all scalars set to zero.
  ///
  /// - Parameter shape: Shape of the tensor.
  @inlinable
  public init(zeros shape: TensorShape, on device: Device = .default) {
    self.init(repeating: 0, shape: shape, on: device)
  }

  /// Creates a tensor with all scalars set to one.
  ///
  /// - Parameter shape: Shape of the tensor.
  @inlinable
  public init(ones shape: TensorShape, on device: Device = .default) {
    self.init(repeating: 1, shape: shape, on: device)
  }

  /// Creates a tensor with all scalars set to zero that has the same shape and type as the provided
  /// tensor.
  ///
  /// - Parameter other: Tensor whose shape and data type to use.
  @inlinable
  public init(zerosLike other: Tensor) {
    self = _Raw.zerosLike(other)
  }

  /// Creates a tensor with all scalars set to one that has the same shape and type as the provided
  /// tensor.
  ///
  /// - Parameter other: Tensor whose shape and data type to use.
  @inlinable
  public init(onesLike other: Tensor) {
    self = _Raw.onesLike(other)
  }

  /// Creates a 1-D tensor representing a sequence from a starting value to, but not including,
  /// an end value, stepping by the specified amount.
  ///
  /// - Parameters:
  ///   - start: The starting value to use for the sequence. If the sequence
  ///     contains any values, the first one is `start`.
  ///   - end: An end value to limit the sequence. `end` is never an element of
  ///     the resulting sequence.
  ///   - stride: The amount to step by with each iteration. `stride` must be
  ///     positive.
  @inlinable
  public init(
    rangeFrom start: Scalar, to end: Scalar, stride: Scalar,
    on device: Device = .default
  ) {
    self = _Raw.range(
      start: Tensor(start, on: device), limit: Tensor(end, on: device),
      delta: Tensor(stride, on: device))
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
  public init(rangeFrom start: Tensor<Scalar>, to end: Tensor<Scalar>, stride: Tensor<Scalar>) {
    self = _Raw.range(start: start, limit: end, delta: stride)
  }

  /// Creates a one-hot tensor at given indices. The locations represented by
  /// `indices` take value `onValue` (`1` by default), while all other locations
  /// take value `offValue` (`0` by default). If the input `indices` is rank
  /// `n`, the new tensor will have rank `n+1`. The new axis is created at
  /// dimension `axis` (by default, the new axis is appended at the end).
  ///
  /// If `indices` is a scalar, the new tensor's shape will be a vector of
  /// length `depth`.
  ///
  /// If `indices` is a vector of length `features`, the output shape will be:
  ///     features x depth, if axis == -1
  ///     depth x features, if axis == 0
  ///
  /// If `indices` is a matrix (batch) with shape `[batch, features]`, the
  /// output shape will be:
  ///     batch x features x depth, if axis == -1
  ///     batch x depth x features, if axis == 1
  ///     depth x batch x features, if axis == 0
  ///
  /// - Parameters:
  ///   - indices: A `Tensor` of indices.
  ///   - depth: A scalar defining the depth of the one hot dimension.
  ///   - onValue: A scalar defining the value at the location referred to by
  ///     some index in `indices`.
  ///   - offValue: A scalar defining the value at a location that is not
  ///     referred to by any index in `indices`.
  ///   - axis: The axis to fill. The default is `-1`, a new inner-most axis.
  @inlinable
  public init(
    oneHotAtIndices indices: Tensor<Int32>,
    depth: Int,
    onValue: Scalar = 1,
    offValue: Scalar = 0,
    axis: Int = -1
  ) {
    let device = indices.device
    self = _Raw.oneHot(
      indices: indices,
      depth: Tensor<Int32>(Int32(depth), on: device),
      onValue: Tensor(onValue, on: device),
      offValue: Tensor(offValue, on: device),
      axis: Int64(axis))
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  /// Creates a 1-D tensor representing a sequence from a starting value, up to and
  /// including an end value, spaced evenly to generate the number of values specified.
  ///
  /// - Parameters:
  ///   - start: The starting value to use for the sequence. If the sequence contains any values,
  ///     the first one is `start`.
  ///   - end: An end value to limit the sequence. `end` is the last element of the resulting
  ///     sequence.
  ///   - count: The number of values in the resulting sequence. `count` must be positive.
  @inlinable
  public init(
    linearSpaceFrom start: Scalar, to end: Scalar, count: Int, on device: Device = .default
  ) {
    self = _Raw.linSpace(
      start: Tensor(start, on: device), stop: Tensor(end, on: device),
      num: Tensor<Int32>(Int32(count), on: device))
  }

  /// Creates a 1-D tensor representing a sequence from a starting value, up to and
  /// including an end value, spaced evenly to generate the number of values specified.
  ///
  /// - Parameters:
  ///   - start: The starting value to use for the sequence. If the sequence contains any values,
  ///     the first one is `start`.
  ///   - end: An end value to limit the sequence. `end` is the last element of the resulting
  ///     sequence.
  ///   - count: The number of values in the resulting sequence. `count` must be positive.
  ///
  /// - Precondition: `start`, `to`, and `count` must be Tensors containing a single Scalar value.
  @inlinable
  public init(linearSpaceFrom start: Tensor<Scalar>, to end: Tensor<Scalar>, count: Tensor<Int32>) {
    self = _Raw.linSpace(start: start, stop: end, num: count)
  }
}

//===------------------------------------------------------------------------------------------===//
// Random
//===------------------------------------------------------------------------------------------===//

extension Tensor where Scalar: TensorFlowIndex {
  /// Creates a tensor with the specified shape, randomly sampling scalar values from a uniform 
  /// distribution between `lowerBound` and `upperBound`.
  ///
  /// - Parameters:
  ///   - shape: The dimensions of the tensor.
  ///   - lowerBound: The lower bound of the distribution.
  ///   - upperBound: The upper bound of the distribution.
  ///   - seed: The seed value.
  public init(
    randomUniform shape: TensorShape,
    lowerBound: Tensor<Scalar>? = nil,
    upperBound: Tensor<Scalar>? = nil,
    seed: TensorFlowSeed = Context.local.randomSeed,
    on device: Device = .default
  ) {
    let lowerBound = lowerBound ?? Tensor<Scalar>(0, on: device)
    let upperBound = upperBound ?? Tensor<Scalar>(1, on: device)
    self = _Raw.statelessRandomUniformInt(
      shape: Tensor<Int32>((0..<shape.rank).map { Int32(shape[$0]) }, on: device),
      seed: Tensor<Int32>([seed.graph, seed.op], on: device),
      minval: lowerBound,
      maxval: upperBound)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  /// Creates a tensor with the specified shape, randomly sampling scalar values from a uniform 
  /// distribution between `lowerBound` and `upperBound`.
  ///
  /// - Parameters:
  ///   - shape: The dimensions of the tensor.
  ///   - lowerBound: The lower bound of the distribution.
  ///   - upperBound: The upper bound of the distribution.
  ///   - seed: The seed value.
  public init(
    randomUniform shape: TensorShape,
    lowerBound: Tensor<Scalar>? = nil,
    upperBound: Tensor<Scalar>? = nil,
    seed: TensorFlowSeed = Context.local.randomSeed,
    on device: Device = .default
  ) {
    let lowerBound = lowerBound ?? Tensor<Scalar>(0, on: device)
    let upperBound = upperBound ?? Tensor<Scalar>(1, on: device)
    let sample: Tensor<Scalar> = _Raw.statelessRandomUniform(
      shape: Tensor<Int32>((0..<shape.rank).map { Int32(shape[$0]) }, on: device),
      seed: Tensor<Int32>([seed.graph, seed.op], on: device))
    self = (upperBound - lowerBound) * sample + lowerBound
  }

  /// Creates a tensor with the specified shape, randomly sampling scalar values from a normal 
  /// distribution.
  ///
  /// - Parameters:
  ///   - shape: The dimensions of the tensor.
  ///   - mean: The mean of the distribution.
  ///   - standardDeviation: The standard deviation of the distribution.
  ///   - seed: The seed value.
  public init(
    randomNormal shape: TensorShape,
    mean: Tensor<Scalar>? = nil,
    standardDeviation: Tensor<Scalar>? = nil,
    seed: TensorFlowSeed = Context.local.randomSeed,
    on device: Device = .default
  ) {
    let sample: Tensor<Scalar> = _Raw.statelessRandomNormal(
      shape: Tensor<Int32>((0..<shape.rank).map { Int32(shape[$0]) }, on: device),
      seed: Tensor<Int32>([seed.graph, seed.op], on: device))
    self =
      (standardDeviation ?? Tensor<Scalar>(1, on: device)) * sample
      + (mean ?? Tensor<Scalar>(0, on: device))
  }

  /// Creates a tensor with the specified shape, randomly sampling scalar values from a truncated
  /// Normal distribution.
  ///
  /// - Parameters:
  ///   - shape: The dimensions of the tensor.
  ///   - mean: The mean of the distribution.
  ///   - standardDeviation: The standard deviation of the distribution.
  ///   - seed: The seed value.
  public init(
    randomTruncatedNormal shape: TensorShape,
    mean: Tensor<Scalar>? = nil,
    standardDeviation: Tensor<Scalar>? = nil,
    seed: TensorFlowSeed = Context.local.randomSeed,
    on device: Device = .default
  ) {
    let sample: Tensor<Scalar> = _Raw.statelessTruncatedNormal(
      shape: Tensor<Int32>((0..<shape.rank).map { Int32(shape[$0]) }, on: device),
      seed: Tensor<Int32>([seed.graph, seed.op], on: device))
    self =
      (standardDeviation ?? Tensor<Scalar>(1, on: device)) * sample
      + (mean ?? Tensor<Scalar>(0, on: device))
  }
}

extension Tensor where Scalar: TensorFlowIndex {
  /// Creates a tensor by drawing samples from a categorical distribution.
  ///
  /// - Parameters:
  ///     - randomCategorialLogits: 2-D Tensor with shape `[batchSize, classCount]`.  Each slice `[i, :]`
  ///         represents the unnormalized log probabilities for all classes.
  ///     - sampleCount: 0-D.  Number of independent samples to draw for each row slice.
  ///     - seed: The seed value.
  ///
  /// - Returns: 2-D Tensor with shape `[batchSize, sampleCount]`.  Each slice `[i, :]`
  ///     contains the drawn class labels with range `[0, classCount)`.
  public init<T: TensorFlowFloatingPoint>(
    randomCategorialLogits: Tensor<T>,
    sampleCount: Int32,
    seed: TensorFlowSeed = Context.local.randomSeed
  ) {
    let device = randomCategorialLogits.device
    self = _Raw.statelessMultinomial(
      logits: randomCategorialLogits,
      numSamples: Tensor<Int32>(sampleCount, on: device),
      seed: Tensor<Int32>([seed.graph, seed.op], on: device))
  }
}

//===------------------------------------------------------------------------------------------===//
// Variance Scaling
//===------------------------------------------------------------------------------------------===//

extension TensorShape {
  // Returns the `fanIn` and `fanOut` counts for `TensorShape`s where the last two axes represent
  // the input channel count and output channel count, respectively.
  fileprivate func fans() -> (in: Int, out: Int) {
    precondition(
      count > 1,
      "Fans cannot be computed for tensors with fewer than 2 dimensions. Got: \(count)")

    // Fans for a 2-D tensor, e.g. `Dense`/`Embedding` weights.
    if count == 2 {
      return (self[0], self[1])
    }
    // Fans for tensors with rank greater than `2`, specifically convolution filters.
    let lastSpatialAxis = endIndex - 3
    let spatialSize = self[0..<(lastSpatialAxis + 1)].contiguousSize
    let inputAxis = endIndex - 2
    let fanIn = self[inputAxis] * spatialSize
    let outputAxis = endIndex - 1
    let fanOut = self[outputAxis] * spatialSize
    return (fanIn, fanOut)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  /// Creates a tensor with the specified shape by performing Glorot (Xavier) uniform initialization.
  ///
  /// It draws random samples from a uniform distribution between `-limit` and `limit`
  /// generated by the default random number generator, where `limit` is
  /// `sqrt(6 / (fanIn + fanOut))` and `fanIn`/`fanOut` represent the number of input and output
  /// features multiplied by the receptive field size.
  ///
  /// Reference: ["Understanding the difficulty of training deep feedforward neural networks"](
  /// http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
  ///
  /// - Parameters:
  ///   - shape: The dimensions of the tensor.
  ///   - seed: The seed value.
  public init(
    glorotUniform shape: TensorShape, seed: TensorFlowSeed = Context.local.randomSeed,
    on device: Device = .default
  ) {
    let (fanIn, fanOut) = shape.fans()
    let limit = Tensor<Scalar>(Scalar.sqrt(6 / Scalar(fanIn + fanOut)), on: device)
    self.init(randomUniform: shape, lowerBound: -limit, upperBound: limit, seed: seed, on: device)
  }

  /// Creates a tensor with the specified shape by performing Glorot (Xavier) normal initialization.
  ///
  /// It draws random samples from a truncated normal distribution centered on `0` with
  /// standard deviation `sqrt(2 / (fanIn + fanOut))` generated by the default random number
  /// generator, where `fanIn`/`fanOut` represent the number of input and output features
  /// multiplied by the receptive field size.
  ///
  /// Reference: ["Understanding the difficulty of training deep feedforward neural networks"](
  /// http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
  ///
  /// - Parameters:
  ///   - shape: The dimensions of the tensor.
  ///   - seed: The seed value.
  public init(
    glorotNormal shape: TensorShape, seed: TensorFlowSeed = Context.local.randomSeed,
    on device: Device = .default
  ) {
    let (fanIn, fanOut) = shape.fans()
    var standardDeviation = Tensor<Scalar>(Scalar.sqrt(2 / Scalar(fanIn + fanOut)), on: device)
    // Standard deviation of truncated standard normal between `-2` and `2` standard deviations.
    let truncationDeviation = Tensor<Scalar>(0.87962566103423978, on: device)
    standardDeviation /= truncationDeviation  // Smooths the tails of the clipped normal.
    self.init(
      randomTruncatedNormal: shape,
      mean: Tensor<Scalar>(0, on: device),
      standardDeviation: standardDeviation,
      seed: seed, on: device)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  /// Creates a tensor with the specified shape by performing He (Kaiming) uniform initialization.
  ///
  /// It draws random samples from a uniform distribution between `-limit` and `limit`
  /// generated by the default random number generator, where `limit` is
  /// `sqrt(6 / fanIn)` and `fanIn` represents the number of input features multiplied by the 
  /// receptive field size.
  ///
  /// Reference: ["Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet 
  /// Classification"](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf)
  ///
  /// - Parameters:
  ///   - shape: The dimensions of the tensor.
  ///   - seed: The seed value.
  public init(
    heUniform shape: TensorShape, seed: TensorFlowSeed = Context.local.randomSeed,
    on device: Device = .default
  ) {
    let (fanIn, _) = shape.fans()
    let limit = Tensor<Scalar>(Scalar.sqrt(6 / Scalar(fanIn)), on: device)
    self.init(randomUniform: shape, lowerBound: -limit, upperBound: limit, seed: seed, on: device)
  }

  /// Creates a tensor with the specified shape by performing He (Kaiming) normal initialization.
  ///
  /// It draws random samples from a truncated normal distribution centered on `0` with
  /// standard deviation `sqrt(2 / fanIn))` generated by the default random number
  /// generator, where `fanIn` represents the number of input features multiplied by the 
  /// receptive field size.
  ///
  /// Reference: ["Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet 
  /// Classification"](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf)
  ///
  /// - Parameters:
  ///   - shape: The dimensions of the tensor.
  ///   - seed: The seed value.
  public init(
    heNormal shape: TensorShape, seed: TensorFlowSeed = Context.local.randomSeed,
    on device: Device = .default
  ) {
    let (fanIn, _) = shape.fans()
    var standardDeviation = Tensor<Scalar>(Scalar.sqrt(2 / Scalar(fanIn)), on: device)
    // Standard deviation of truncated standard normal between `-2` and `2` standard deviations.
    let truncationDeviation = Tensor<Scalar>(0.87962566103423978, on: device)
    standardDeviation /= truncationDeviation  // Smooths the tails of the clipped normal.
    self.init(
      randomTruncatedNormal: shape,
      mean: Tensor<Scalar>(0, on: device),
      standardDeviation: standardDeviation,
      seed: seed, on: device)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  /// Creates a tensor with the specified shape by performing LeCun uniform initialization.
  ///
  /// It draws random samples from a uniform distribution between `-limit` and `limit`
  /// generated by the default random number generator, where `limit` is
  /// `sqrt(3 / fanIn)` and `fanIn` represents the number of input features multiplied
  /// by the receptive field size.
  ///
  /// Reference: ["Efficient BackProp"](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
  ///
  /// - Parameters:
  ///   - shape: The dimensions of the tensor.
  ///   - seed: The seed value.
  public init(
    leCunUniform shape: TensorShape, seed: TensorFlowSeed = Context.local.randomSeed,
    on device: Device = .default
  ) {
    let (fanIn, _) = shape.fans()
    let limit = Tensor<Scalar>(Scalar.sqrt(3 / Scalar(fanIn)), on: device)
    self.init(randomUniform: shape, lowerBound: -limit, upperBound: limit, seed: seed, on: device)
  }

  /// Creates a tensor with the specified shape by performing LeCun normal initialization.
  ///
  /// It draws random samples from a truncated normal distribution centered on `0` with
  /// standard deviation `sqrt(1 / fanIn)` generated by the default random number
  /// generator, where `fanIn` represents the number of input features multiplied by the 
  /// receptive field size.
  ///
  /// Reference: ["Efficient BackProp"](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
  ///
  /// - Parameters:
  ///   - shape: The dimensions of the tensor.
  ///   - seed: The seed value.
  public init(
    leCunNormal shape: TensorShape, seed: TensorFlowSeed = Context.local.randomSeed,
    on device: Device = .default
  ) {
    let (fanIn, _) = shape.fans()
    var standardDeviation = Tensor<Scalar>(Scalar.sqrt(1 / Scalar(fanIn)), on: device)
    // Standard deviation of truncated standard normal between `-2` and `2` standard deviations.
    let truncationDeviation = Tensor<Scalar>(0.87962566103423978, on: device)
    standardDeviation /= truncationDeviation  // Smooths the tails of the clipped normal.
    self.init(
      randomTruncatedNormal: shape,
      mean: Tensor<Scalar>(0, on: device),
      standardDeviation: standardDeviation,
      seed: seed, on: device)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  /// Creates an orthogonal matrix or tensor. 
  ///
  /// If the shape of the tensor to initialize is two-dimensional, it is initialized with an 
  /// orthogonal matrix obtained from the QR decomposition of a matrix of random numbers drawn 
  /// from a normal distribution. If the matrix has fewer rows than columns then the output will 
  /// have orthogonal rows. Otherwise, the output will have orthogonal columns.
  /// 
  /// If the shape of the tensor to initialize is more than two-dimensional, a matrix of shape 
  /// `[shape[0] * ... * shape[rank - 2], shape[rank - 1]]` is initialized.  The matrix is 
  /// subsequently reshaped to give a tensor of the desired shape.
  ///
  /// - Parameters:
  ///   - shape: The shape of the tensor.
  ///   - gain: A multiplicative factor to apply to the orthogonal tensor.
  ///   - seed: A tuple of two integers to seed the random number generator.
  public init(
    orthogonal shape: TensorShape,
    gain: Tensor<Scalar> = Tensor<Scalar>(1),
    seed: TensorFlowSeed = Context.local.randomSeed
  ) {
    let rowCount = shape.dimensions.dropLast().reduce(1, *)
    let columnCount = shape[shape.rank - 1]
    var flatShape: TensorShape
    if rowCount < columnCount {
      flatShape = [columnCount, rowCount]
    } else {
      flatShape = [rowCount, columnCount]
    }
    let normal = Tensor(randomNormal: flatShape, seed: seed)
    var (q, r) = normal.qrDecomposition(fullMatrices: false)
    let d = r.diagonalPart()
    q *= sign(d)
    if rowCount < columnCount {
      q = q.transposed()
    }
    self = q.reshaped(to: shape) * gain
  }
}
