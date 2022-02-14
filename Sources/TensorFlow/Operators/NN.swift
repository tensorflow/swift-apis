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

//===------------------------------------------------------------------------------------------===//
// Normalization
//===------------------------------------------------------------------------------------------===//

extension Tensor where Scalar: TensorFlowFloatingPoint {
  /// Returns a tensor computed from batch-normalizing the input along the specified axis.
  ///
  /// Specifically, returns `(self - mu) / (var + epsilon) * gamma + beta` where `mu` and `var`
  /// are respectively the mean and variance of `self` along `axis`.
  ///
  /// - Parameters:
  ///   - axis: The batch dimension.
  ///   - offset: The offset, also known as beta.
  ///   - scale: The scale, also known as gamma.
  ///   - epsilon: A small value added to the denominator for numerical stability.
  @inlinable
  @differentiable(reverse, wrt: (self, offset, scale))
  public func batchNormalized(
    alongAxis axis: Int,
    offset: Tensor = Tensor(0),
    scale: Tensor = Tensor(1),
    epsilon: Scalar = 0.001
  ) -> Tensor {
    let moments = self.moments(alongAxes: axis)
    let inv = rsqrt(moments.variance + epsilon) * scale
    return self * inv + offset - moments.mean * inv
  }
}

//===------------------------------------------------------------------------------------------===//
// Convolution and Pooling
//===------------------------------------------------------------------------------------------===//

/// A padding scheme. Used by padding, convolution, and pooling ops.
// @_frozen // SR-9739
public enum Padding {
  /// The "valid" padding scheme.
  case valid
  /// The "same" padding scheme.
  case same
}

extension Padding {
  @inlinable
  public var raw: _Raw.Padding {
    switch self {
    case .same: return .same
    case .valid: return .valid
    }
  }

  @inlinable
  internal var raw2: _Raw.Padding1 {
    switch self {
    case .same: return .same
    case .valid: return .valid
    }
  }
}

/// Returns a 1-D convolution with the specified input, filter, stride, and padding.
///
/// - Parameters:
///   - input: The input.
///   - filter: The convolution filter.
///   - stride: The stride of the sliding filter.
///   - padding: The padding for the operation.
///   - dilation: The dilation factor.
/// - Precondition: `input` must have rank `3`.
/// - Precondition: `filter` must have rank 3.
@differentiable(reverse, wrt: (input, filter))
public func conv1D<Scalar: TensorFlowFloatingPoint>(
  _ input: Tensor<Scalar>,
  filter: Tensor<Scalar>,
  stride: Int = 1,
  padding: Padding = .valid,
  dilation: Int = 1
) -> Tensor<Scalar> {
  precondition(input.shape.rank == 3, "The input must have rank 3.")
  precondition(filter.shape.rank == 3, "The filter must have rank 3.")
  return conv2D(
    input.expandingShape(at: 1),
    filter: filter.expandingShape(at: 0),
    strides: (1, 1, stride, 1),
    padding: padding,
    dilations: (1, 1, dilation, 1)
  ).squeezingShape(at: 1)
}

/// Returns a 2-D convolution with the specified input, filter, strides, and padding.
///
/// - Parameters:
///   - input: The input.
///   - filter: The convolution filter.
///   - strides: The strides of the sliding filter for each dimension of the input.
///   - padding: The padding for the operation
///   - dilations: The dilation factor for each dimension of the input.
/// - Precondition: `input` must have rank `4`.
/// - Precondition: `filter` must have rank 4.
@differentiable(reverse, wrt: (input, filter))
public func conv2D<Scalar: TensorFlowFloatingPoint>(
  _ input: Tensor<Scalar>,
  filter: Tensor<Scalar>,
  strides: (Int, Int, Int, Int) = (1, 1, 1, 1),
  padding: Padding = .valid,
  dilations: (Int, Int, Int, Int) = (1, 1, 1, 1)
) -> Tensor<Scalar> {
  precondition(input.shape.rank == 4, "The input must have rank 4.")
  precondition(filter.shape.rank == 4, "The filter must have rank 4.")
  return _Raw.conv2D(
    input,
    filter: filter,
    strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3)],
    padding: padding.raw2,
    explicitPaddings: [],
    dilations: [Int32(dilations.0), Int32(dilations.1), Int32(dilations.2), Int32(dilations.3)]
  )
}

@usableFromInline
@derivative(of: conv2D)
func _vjpConv2D<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  filter: Tensor<Scalar>,
  strides: (Int, Int, Int, Int),
  padding: Padding,
  dilations: (Int, Int, Int, Int)
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
  let value = conv2D(x, filter: filter, strides: strides, padding: padding, dilations: dilations)
  return (
    value,
    { v in
      (
        conv2DBackpropInput(
          v, shape: x.shape.dimensions.map { Int64($0) }, filter: filter,
          strides: strides, padding: padding, dilations: dilations),
        conv2DBackpropFilter(
          v, input: x, filterSizes: filter.shape.dimensions.map { Int64($0) },
          strides: strides, padding: padding, dilations: dilations)
      )
    }
  )
}

/// Returns a 2-D transposed convolution with the specified input, filter, strides, and padding.
///
/// - Parameters:
///   - input: The input.
///   - shape: The output shape of the deconvolution operation.
///   - filter: The convolution filter.
///   - strides: The strides of the sliding filter for each dimension of the input.
///   - padding: The padding for the operation
///   - dilations: The dilation factor for each dimension of the input.
/// - Precondition: `input` must have rank `4`.
/// - Precondition: `filter` must have rank 4.
@differentiable(reverse, wrt: (input, filter))
public func transposedConv2D<Scalar: TensorFlowFloatingPoint>(
  _ input: Tensor<Scalar>,
  shape: [Int64],
  filter: Tensor<Scalar>,
  strides: (Int, Int, Int, Int) = (1, 1, 1, 1),
  padding: Padding = .valid,
  dilations: (Int, Int, Int, Int) = (1, 1, 1, 1)
) -> Tensor<Scalar> {
  precondition(input.shape.rank == 4, "The input must have rank 4.")
  precondition(filter.shape.rank == 4, "The filter must have rank 4.")
  return conv2DBackpropInput(
    input, shape: shape, filter: filter,
    strides: strides, padding: padding, dilations: dilations)
}

/// TensorFlow builtin conv2d gradient helper for the input.
@differentiable(reverse, wrt: (x, filter))
@usableFromInline
func conv2DBackpropInput<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  shape: [Int64],
  filter: Tensor<Scalar>,
  strides: (Int, Int, Int, Int) = (1, 1, 1, 1),
  padding: Padding = .valid,
  dilations: (Int, Int, Int, Int) = (1, 1, 1, 1)
) -> Tensor<Scalar> {
  return _Raw.conv2DBackpropInput(
    inputSizes: shape,
    filter: filter,
    outBackprop: x,
    strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3)],
    padding: padding.raw2,
    explicitPaddings: [],
    dilations: [Int32(dilations.0), Int32(dilations.1), Int32(dilations.2), Int32(dilations.3)])
}

@derivative(of: conv2DBackpropInput)
@usableFromInline
func _vjpConv2DBackpropInput<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  _ shape: [Int64],
  _ filter: Tensor<Scalar>,
  _ strides: (Int, Int, Int, Int),
  _ padding: Padding,
  _ dilations: (Int, Int, Int, Int)
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
  let value = conv2DBackpropInput(
    x, shape: shape, filter: filter,
    strides: strides, padding: padding, dilations: dilations)
  return (
    value,
    { v in
      (
        conv2D(v, filter: filter, strides: strides, padding: padding, dilations: dilations),
        conv2DBackpropFilter(
          x, input: v, filterSizes: filter.shape.dimensions.map { Int64($0) }, strides: strides,
          padding: padding, dilations: dilations)
      )
    }
  )
}

/// TensorFlow builtin conv2d gradient helper for the filter.
@differentiable(reverse, wrt: (x, input))
@usableFromInline
func conv2DBackpropFilter<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  input: Tensor<Scalar>,
  filterSizes: [Int64],
  strides: (Int, Int, Int, Int) = (1, 1, 1, 1),
  padding: Padding = .valid,
  dilations: (Int, Int, Int, Int) = (1, 1, 1, 1)
) -> Tensor<Scalar> {
  return _Raw.conv2DBackpropFilter(
    input,
    filterSizes: filterSizes,
    outBackprop: x,
    strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3)],
    padding: padding.raw2,
    explicitPaddings: [],
    dilations: [Int32(dilations.0), Int32(dilations.1), Int32(dilations.2), Int32(dilations.3)])
}

@usableFromInline
@derivative(of: conv2DBackpropFilter)
func _vjpConv2DBackpropFilter<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  _ input: Tensor<Scalar>,
  _ filterSizes: [Int64],
  _ strides: (Int, Int, Int, Int),
  _ padding: Padding,
  _ dilations: (Int, Int, Int, Int)
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
  let value = conv2DBackpropFilter(
    x, input: input, filterSizes: filterSizes,
    strides: strides, padding: padding, dilations: dilations)
  return (
    value,
    { v in
      (
        conv2D(input, filter: v, strides: strides, padding: padding, dilations: dilations),
        conv2DBackpropInput(
          x, shape: x.shape.dimensions.map { Int64($0) }, filter: v, strides: strides,
          padding: padding, dilations: dilations)
      )
    }
  )
}

/// Returns a 3-D convolution with the specified input, filter, strides, padding and dilations.
///
/// - Parameters:
///   - input: The input.
///   - filter: The convolution filter.
///   - strides: The strides of the sliding filter for each dimension of the input.
///   - padding: The padding for the operation.
///   - dilations: The dilation factor for each dimension of the input.
/// - Precondition: `input` must have rank `5`.
/// - Precondition: `filter` must have rank 5.
@differentiable(reverse, wrt: (input, filter))
public func conv3D<Scalar: TensorFlowFloatingPoint>(
  _ input: Tensor<Scalar>,
  filter: Tensor<Scalar>,
  strides: (Int, Int, Int, Int, Int) = (1, 1, 1, 1, 1),
  padding: Padding = .valid,
  dilations: (Int, Int, Int, Int, Int) = (1, 1, 1, 1, 1)
) -> Tensor<Scalar> {
  precondition(input.shape.rank == 5, "The input must have rank 5.")
  precondition(filter.shape.rank == 5, "The filter must have rank 5.")
  return _Raw.conv3D(
    input,
    filter: filter,
    strides: [
      Int32(strides.0), Int32(strides.1), Int32(strides.2),
      Int32(strides.3), Int32(strides.4),
    ],
    padding: padding.raw,
    dilations: [
      Int32(dilations.0), Int32(dilations.1), Int32(dilations.2),
      Int32(dilations.3), Int32(dilations.4),
    ]
  )
}

@usableFromInline
@derivative(of: conv3D)
func _vjpConv3D<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  filter: Tensor<Scalar>,
  strides: (Int, Int, Int, Int, Int),
  padding: Padding,
  dilations: (Int, Int, Int, Int, Int)
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
  let value = conv3D(
    x, filter: filter, strides: strides,
    padding: padding, dilations: dilations)
  return (
    value,
    { v in
      (
        conv3DBackpropInput(
          v, shape: x.shapeTensor, filter: filter,
          strides: strides, padding: padding),
        conv3DBackpropFilter(
          v, input: x, filterSizes: filter.shapeTensor,
          strides: strides, padding: padding)
      )
    }
  )
}

/// TensorFlow builtin conv3d gradient helper for the input.
@differentiable(reverse, wrt: (x, filter))
@usableFromInline
func conv3DBackpropInput<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  shape: Tensor<Int32>,
  filter: Tensor<Scalar>,
  strides: (Int, Int, Int, Int, Int) = (1, 1, 1, 1, 1),
  padding: Padding = .valid,
  dilations: (Int, Int, Int, Int, Int) = (1, 1, 1, 1, 1)
) -> Tensor<Scalar> {
  return _Raw.conv3DBackpropInputV2(
    inputSizes: shape,
    filter: filter,
    outBackprop: x,
    strides: [
      Int32(strides.0), Int32(strides.1), Int32(strides.2),
      Int32(strides.3), Int32(strides.4),
    ],
    padding: padding.raw,
    dilations: [
      Int32(dilations.0), Int32(dilations.1), Int32(dilations.2),
      Int32(dilations.3), Int32(dilations.4),
    ]
  )
}

@usableFromInline
@derivative(of: conv3DBackpropInput)
func _vjpConv3DBackpropInput<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  _ shape: Tensor<Int32>,
  _ filter: Tensor<Scalar>,
  _ strides: (Int, Int, Int, Int, Int),
  _ padding: Padding,
  _ dilations: (Int, Int, Int, Int, Int)
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
  let value = conv3DBackpropInput(
    x, shape: shape, filter: filter, strides: strides,
    padding: padding, dilations: dilations)
  return (
    value,
    { v in
      (
        conv3D(v, filter: filter, strides: strides, padding: padding),
        conv3DBackpropFilter(
          x, input: v, filterSizes: filter.shapeTensor, strides: strides,
          padding: padding)
      )
    }
  )
}

/// TensorFlow builtin conv3d gradient helper for the filter.
@differentiable(reverse, wrt: (x, input))
@usableFromInline
func conv3DBackpropFilter<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  input: Tensor<Scalar>,
  filterSizes: Tensor<Int32>,
  strides: (Int, Int, Int, Int, Int) = (1, 1, 1, 1, 1),
  padding: Padding = .valid,
  dilations: (Int, Int, Int, Int, Int) = (1, 1, 1, 1, 1)
) -> Tensor<Scalar> {
  return _Raw.conv3DBackpropFilterV2(
    input,
    filterSizes: filterSizes,
    outBackprop: x,
    strides: [
      Int32(strides.0), Int32(strides.1), Int32(strides.2),
      Int32(strides.3), Int32(strides.4),
    ],
    padding: padding.raw,
    dilations: [
      Int32(dilations.0), Int32(dilations.1), Int32(dilations.2),
      Int32(dilations.3), Int32(dilations.4),
    ]
  )
}

@usableFromInline
@derivative(of: conv3DBackpropFilter)
func _vjpConv3DBackpropFilter<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  _ input: Tensor<Scalar>,
  _ filterSizes: Tensor<Int32>,
  _ strides: (Int, Int, Int, Int, Int),
  _ padding: Padding,
  _ dilations: (Int, Int, Int, Int, Int)
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
  let value = conv3DBackpropFilter(
    x, input: input, filterSizes: filterSizes,
    strides: strides, padding: padding, dilations: dilations)
  return (
    value,
    { v in
      (
        conv3D(input, filter: v, strides: strides, padding: padding),
        conv3DBackpropInput(
          x, shape: x.shapeTensor, filter: v, strides: strides,
          padding: padding)
      )
    }
  )
}

/// Returns a 2-D depthwise convolution with the specified input, filter, strides, and padding.
///
/// - Parameters:
///   - input: The input.
///   - filter: The depthwise convolution filter.
///   - strides: The strides of the sliding filter for each dimension of the input.
///   - padding: The padding for the operation.
/// - Precondition: `input` must have rank 4.
/// - Precondition: `filter` must have rank 4.
@differentiable(reverse, wrt: (input, filter))
public func depthwiseConv2D<Scalar: TensorFlowFloatingPoint>(
  _ input: Tensor<Scalar>,
  filter: Tensor<Scalar>,
  strides: (Int, Int, Int, Int) = (1, 1, 1, 1),
  padding: Padding,
  dilations: (Int, Int, Int, Int) = (1, 1, 1, 1)
) -> Tensor<Scalar> {
  precondition(input.shape.rank == 4, "The input must have rank 4.")
  precondition(filter.shape.rank == 4, "The filter must have rank 4.")
  return _Raw.depthwiseConv2dNative(
    input,
    filter: filter,
    strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3)],
    padding: padding.raw,
    dilations: [Int32(dilations.0), Int32(dilations.1), Int32(dilations.2), Int32(dilations.3)])
}

@usableFromInline
@derivative(of: depthwiseConv2D)
func _vjpDepthwiseConv2D<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  filter: Tensor<Scalar>,
  strides: (Int, Int, Int, Int),
  padding: Padding,
  dilations: (Int, Int, Int, Int)
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
  let value = depthwiseConv2D(
    x, filter: filter, strides: strides,
    padding: padding, dilations: dilations)
  return (
    value,
    { v in
      (
        depthwiseConv2dBackpropInput(
          v, shape: x.shapeTensor, filter: filter,
          strides: strides, padding: padding, dilations: dilations),
        depthwiseConv2dBackpropFilter(
          v, input: x, filterSizes: filter.shapeTensor,
          strides: strides, padding: padding, dilations: dilations)
      )
    }
  )
}

/// TensorFlow builtin depthwiseConv2D gradient helper for the input.
@differentiable(reverse, wrt: (x, filter))
@usableFromInline
func depthwiseConv2dBackpropInput<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  shape: Tensor<Int32>,
  filter: Tensor<Scalar>,
  strides: (Int, Int, Int, Int) = (1, 1, 1, 1),
  padding: Padding,
  dilations: (Int, Int, Int, Int) = (1, 1, 1, 1)
) -> Tensor<Scalar> {
  return _Raw.depthwiseConv2dNativeBackpropInput(
    inputSizes: shape,
    filter: filter,
    outBackprop: x,
    strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3)],
    padding: padding.raw,
    dilations: [Int32(dilations.0), Int32(dilations.1), Int32(dilations.2), Int32(dilations.3)])
}

@usableFromInline
@derivative(of: depthwiseConv2dBackpropInput)
func _vjpDepthwiseConv2dBackpropInput<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  _ shape: Tensor<Int32>,
  _ filter: Tensor<Scalar>,
  _ strides: (Int, Int, Int, Int),
  _ padding: Padding,
  _ dilations: (Int, Int, Int, Int)
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
  let value = depthwiseConv2dBackpropInput(
    x, shape: shape, filter: filter, strides: strides,
    padding: padding, dilations: dilations)
  return (
    value,
    { v in
      (
        depthwiseConv2D(v, filter: filter, strides: strides, padding: padding, dilations: dilations),
        depthwiseConv2dBackpropFilter(
          x, input: v, filterSizes: filter.shapeTensor,
          strides: strides, padding: padding, dilations: dilations)
      )

    }
  )
}

/// TensorFlow builtin depthwiseConv2D gradient helper for the filter.
@differentiable(reverse, wrt: (x, input))
@usableFromInline
func depthwiseConv2dBackpropFilter<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  input: Tensor<Scalar>,
  filterSizes: Tensor<Int32>,
  strides: (Int, Int, Int, Int) = (1, 1, 1, 1),
  padding: Padding,
  dilations: (Int, Int, Int, Int) = (1, 1, 1, 1)
) -> Tensor<Scalar> {
  return _Raw.depthwiseConv2dNativeBackpropFilter(
    input,
    filterSizes: filterSizes,
    outBackprop: x,
    strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3)],
    padding: padding.raw,
    dilations: [Int32(dilations.0), Int32(dilations.1), Int32(dilations.2), Int32(dilations.3)])
}

@usableFromInline
@derivative(of: depthwiseConv2dBackpropFilter)
func _vjpDepthwiseConv2dBackpropFilter<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  _ input: Tensor<Scalar>,
  _ filterSizes: Tensor<Int32>,
  _ strides: (Int, Int, Int, Int),
  _ padding: Padding,
  _ dilations: (Int, Int, Int, Int)
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
  let value = depthwiseConv2dBackpropFilter(
    x, input: input, filterSizes: filterSizes,
    strides: strides, padding: padding, dilations: dilations)
  return (
    value,
    { v in
      (
        depthwiseConv2D(input, filter: v, strides: strides, padding: padding, dilations: dilations),
        depthwiseConv2dBackpropInput(
          x, shape: x.shapeTensor, filter: v, strides: strides,
          padding: padding, dilations: dilations)
      )
    }
  )
}

/// Returns a 2-D max pooling, with the specified filter sizes, strides, and
/// padding.
///
/// - Parameters:
///   - input: The input.
///   - filterSize: The dimensions of the pooling kernel.
///   - strides: The strides of the sliding filter for each dimension of the input.
///   - padding: The padding for the operation.
@differentiable(reverse, wrt: input)
public func maxPool2D<Scalar: TensorFlowFloatingPoint>(
  _ input: Tensor<Scalar>,
  filterSize: (Int, Int, Int, Int),
  strides: (Int, Int, Int, Int),
  padding: Padding
) -> Tensor<Scalar> {
  precondition(input.rank == 4, "The rank of the input must be 4.")
  return _Raw.maxPoolV2(
    input,
    ksize: [
      Int64(filterSize.0), Int64(filterSize.1),
      Int64(filterSize.2), Int64(filterSize.3),
    ],
    strides: [
      Int64(strides.0), Int64(strides.1),
      Int64(strides.2), Int64(strides.3),
    ],
    padding: padding.raw)
}

@usableFromInline
@derivative(of: maxPool2D)
func _vjpMaxPool2D<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  filterSize: (Int, Int, Int, Int),
  strides: (Int, Int, Int, Int),
  padding: Padding
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> Tensor<Scalar>) {
  // TODO: Currently this is not higher order differentiable. Redefine in
  // closed form.
  let value = maxPool2D(x, filterSize: filterSize, strides: strides, padding: padding)
  return (
    value,
    { v in
      _Raw.maxPoolGradV2(
        origInput: x,
        origOutput: value,
        grad: v,
        ksize: [
          Int64(filterSize.0), Int64(filterSize.1),
          Int64(filterSize.2), Int64(filterSize.3),
        ],
        strides: [
          Int64(strides.0), Int64(strides.1),
          Int64(strides.2), Int64(strides.3),
        ],
        padding: padding.raw)
    }
  )
}

/// Returns a 3-D max pooling, with the specified filter sizes, strides, and
/// padding.
///
/// - Parameters:
///   - input: The input.
///   - filterSize: The dimensions of the pooling kernel.
///   - strides: The strides of the sliding filter for each dimension of the input.
///   - padding: The padding for the operation.
@differentiable(reverse, wrt: input)
public func maxPool3D<Scalar: TensorFlowFloatingPoint>(
  _ input: Tensor<Scalar>,
  filterSize: (Int, Int, Int, Int, Int),
  strides: (Int, Int, Int, Int, Int),
  padding: Padding
) -> Tensor<Scalar> {
  precondition(input.rank == 5, "The rank of the input must be 5.")
  return _Raw.maxPool3D(
    input,
    ksize: [
      Int32(filterSize.0), Int32(filterSize.1),
      Int32(filterSize.2), Int32(filterSize.3), Int32(filterSize.4),
    ],
    strides: [
      Int32(strides.0), Int32(strides.1),
      Int32(strides.2), Int32(strides.3), Int32(strides.4),
    ],
    padding: padding.raw)
}

@usableFromInline
@derivative(of: maxPool3D)
func _vjpMaxPool3D<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  filterSize: (Int, Int, Int, Int, Int),
  strides: (Int, Int, Int, Int, Int),
  padding: Padding
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> Tensor<Scalar>) {
  // TODO: Currently this is not higher order differentiable. Redefine in
  // closed form.
  let value = maxPool3D(x, filterSize: filterSize, strides: strides, padding: padding)
  return (
    value,
    { v in
      return _Raw.maxPool3DGrad(
        origInput: x,
        origOutput: value,
        grad: v,
        ksize: [
          Int32(filterSize.0), Int32(filterSize.1), Int32(filterSize.2),
          Int32(filterSize.3), Int32(filterSize.4),
        ],
        strides: [
          Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3),
          Int32(strides.4),
        ],
        padding: padding.raw
      )
    }
  )
}

/// Returns a 2-D average pooling, with the specified filter sizes, strides,
/// and padding.
///
/// - Parameters:
///   - input: The input.
///   - filterSize: The dimensions of the pooling kernel.
///   - strides: The strides of the sliding filter for each dimension of the input.
///   - padding: The padding for the operation.
@differentiable(reverse, wrt: input)
public func avgPool2D<Scalar: TensorFlowFloatingPoint>(
  _ input: Tensor<Scalar>,
  filterSize: (Int, Int, Int, Int),
  strides: (Int, Int, Int, Int),
  padding: Padding
) -> Tensor<Scalar> {
  precondition(input.rank == 4, "The rank of the input must be 4.")
  return _Raw.avgPool(
    value: input,
    ksize: [
      Int32(filterSize.0), Int32(filterSize.1),
      Int32(filterSize.2), Int32(filterSize.3),
    ],
    strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3)],
    padding: padding.raw)
}

@usableFromInline
@derivative(of: avgPool2D)
func _vjpAvgPool2D<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  filterSize: (Int, Int, Int, Int),
  strides: (Int, Int, Int, Int),
  padding: Padding
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> Tensor<Scalar>) {
  // TODO: Currently this is not higher order differentiable. Redefine in
  // closed form.
  let value = avgPool2D(x, filterSize: filterSize, strides: strides, padding: padding)
  return (
    value,
    { v in
      _Raw.avgPoolGrad(
        origInputShape: x.shapeTensor,
        grad: v,
        ksize: [
          Int32(filterSize.0), Int32(filterSize.1),
          Int32(filterSize.2), Int32(filterSize.3),
        ],
        strides: [
          Int32(strides.0), Int32(strides.1),
          Int32(strides.2), Int32(strides.3),
        ],
        padding: padding.raw
      )
    }
  )
}

/// Returns a 3-D average pooling, with the specified filter sizes, strides,
/// and padding.
///
/// - Parameters:
///   - input: The input.
///   - filterSize: The dimensions of the pooling kernel.
///   - strides: The strides of the sliding filter for each dimension of the input.
///   - padding: The padding for the operation.
@differentiable(reverse, wrt: input)
public func avgPool3D<Scalar: TensorFlowFloatingPoint>(
  _ input: Tensor<Scalar>,
  filterSize: (Int, Int, Int, Int, Int),
  strides: (Int, Int, Int, Int, Int),
  padding: Padding
) -> Tensor<Scalar> {
  precondition(input.rank == 5, "The rank of the input must be 5.")
  return _Raw.avgPool3D(
    input,
    ksize: [
      Int32(filterSize.0), Int32(filterSize.1),
      Int32(filterSize.2), Int32(filterSize.3), Int32(filterSize.4),
    ],
    strides: [
      Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3),
      Int32(strides.4),
    ],
    padding: padding.raw)
}

@usableFromInline
@derivative(of: avgPool3D)
func _vjpAvgPool3D<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  filterSize: (Int, Int, Int, Int, Int),
  strides: (Int, Int, Int, Int, Int),
  padding: Padding
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> Tensor<Scalar>) {
  // TODO: Currently this is not higher order differentiable. Redefine in
  // closed form.
  let value = avgPool3D(x, filterSize: filterSize, strides: strides, padding: padding)
  return (
    value,
    { v in
      return _Raw.avgPool3DGrad(
        origInputShape: x.shapeTensor,
        grad: v,
        ksize: [
          Int32(filterSize.0), Int32(filterSize.1), Int32(filterSize.2),
          Int32(filterSize.3), Int32(filterSize.4),
        ],
        strides: [
          Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3),
          Int32(strides.4),
        ],
        padding: padding.raw
      )
    }
  )
}

/// Returns a 2-D fractional max pooling, with the specified pooling ratios.
///
/// Note: `fractionalMaxPool` does not have an XLA implementation, and thus may have performance implications.
///
/// - Parameters:
///   - input: A Tensor. 4-D with shape `[batch, height, width, channels]`.
///   - poolingRatio: A list of `Doubles`. Pooling ratio for each dimension of `input`, currently only
///     supports row and col dimension and should be >= 1.0.
///   - pseudoRandom: An optional `Bool`. Defaults to `false`. When set to `true`,
///     generates the pooling sequence in a pseudorandom fashion, otherwise, in a random fashion.
///   - overlapping: An optional `Bool`. Defaults to `false`. When set to `true`, it means
///     when pooling, the values at the boundary of adjacent pooling cells are used by both cells.
///   - deterministic: An Optional `Bool`. When set to `true`, a fixed pooling region will be
///     used when iterating over a fractionalMaxPool2D node in the computation graph.
///   - seed: An optional `Int64`. Defaults to `0`. If set to be non-zero, the random number
///     generator is seeded by the given seed.
///   - seed2: An optional `Int64`. Defaults to `0`. A second seed to avoid seed collision.
@differentiable(reverse, wrt: input)
public func fractionalMaxPool2D<Scalar: TensorFlowFloatingPoint>(
  _ input: Tensor<Scalar>,
  poolingRatio: (Double, Double, Double, Double),
  pseudoRandom: Bool = false,
  overlapping: Bool = false,
  deterministic: Bool = false,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<Scalar> {
  precondition(input.rank == 4, "The rank of the input must be 4.")
  return _Raw.fractionalMaxPool(
    value: input,
    poolingRatio: [
      Double(poolingRatio.0), Double(poolingRatio.1),
      Double(poolingRatio.2), Double(poolingRatio.3),
    ],
    pseudoRandom: pseudoRandom,
    overlapping: overlapping,
    deterministic: deterministic,
    seed: seed,
    seed2: seed2
  ).0
}

@usableFromInline
@derivative(of: fractionalMaxPool2D)
func _vjpFractionalMaxPool<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  poolingRatio: (Double, Double, Double, Double),
  pseudoRandom: Bool = false,
  overlapping: Bool = false,
  deterministic: Bool = false,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> Tensor<Scalar>) {
  // TODO: Currently this is not higher order differentiable. Redefine in
  // closed form.
  let (value, rowPoolingSequence, colPoolingSequence) = _Raw.fractionalMaxPool(
    value: x,
    poolingRatio: [
      Double(poolingRatio.0), Double(poolingRatio.1),
      Double(poolingRatio.2), Double(poolingRatio.3),
    ],
    pseudoRandom: pseudoRandom,
    overlapping: overlapping,
    deterministic: deterministic,
    seed: seed,
    seed2: seed2)
  return (
    value,
    { v in
      _Raw.fractionalMaxPoolGrad(
        origInput: x,
        origOutput: value,
        outBackprop: v,
        rowPoolingSequence: rowPoolingSequence,
        colPoolingSequence: colPoolingSequence,
        overlapping: overlapping
      )
    }
  )
}

//===------------------------------------------------------------------------------------------===//
// Rearrange depth/space
//===------------------------------------------------------------------------------------------===//

/// Returns a copy of `input` where values from the depth dimension are moved in spatial blocks to the height and width dimensions.
///
/// For example, given an input of shape `[1, 2, 2, 1]`, data_format = "NHWC" and
/// block_size = 2:
///
/// ```
/// x = [[[[1], [2]],
///       [[3], [4]]]]
/// ```
///
/// This operation will output a tensor of shape `[1, 1, 1, 4]`:
///
/// ```
/// [[[[1, 2, 3, 4]]]]
/// ```
///
/// Here, the input has a batch of 1 and each batch element has shape `[2, 2, 1]`,
/// the corresponding output will have a single element (i.e. width and height are
/// both 1) and will have a depth of 4 channels (1 * block_size * block_size).
/// The output element shape is `[1, 1, 4]`.
///
/// For an input tensor with larger depth, here of shape `[1, 2, 2, 3]`, e.g.
///
/// ```
/// x = [[[[1, 2, 3], [4, 5, 6]],
///       [[7, 8, 9], [10, 11, 12]]]]
/// ```
///
/// This operation, for block_size of 2, will return the following tensor of shape
/// `[1, 1, 1, 12]`
///
/// ```
/// [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
/// ```
///
/// Similarly, for the following input of shape `[1 4 4 1]`, and a block size of 2:
///
/// ```
/// x = [[[[1],   [2],  [5],  [6]],
///       [[3],   [4],  [7],  [8]],
///       [[9],  [10], [13],  [14]],
///       [[11], [12], [15],  [16]]]]
/// ```
///
/// the operator will return the following tensor of shape `[1 2 2 4]`:
///
/// ```
/// x = [[[[1, 2, 3, 4],
///        [5, 6, 7, 8]],
///       [[9, 10, 11, 12],
///        [13, 14, 15, 16]]]]
/// ```
///
/// - Precondition: `input.rank == 4 && b >= 2`.
/// - Precondition: The number of the features must be divisible by square of `b`.
@differentiable(reverse, wrt: input where Scalar: TensorFlowFloatingPoint)
public func depthToSpace<Scalar>(_ input: Tensor<Scalar>, blockSize b: Int) -> Tensor<Scalar> {
  precondition(input.rank == 4, "The input must have rank 4.")
  precondition(b >= 2, "The size must be greater than 1.")
  precondition(
    input.shape[3].isMultiple(of: b * b),
    "The number of the features must be divisible by square of the block size.")
  return _Raw.depthToSpace(input, blockSize: Int64(b))
}

@usableFromInline
@derivative(of: depthToSpace)
func _vjpDepthToSpace<Scalar: TensorFlowFloatingPoint>(
  _ input: Tensor<Scalar>,
  blockSize b: Int
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> Tensor<Scalar>) {
  (depthToSpace(input, blockSize: b), { spaceToDepth($0, blockSize: b) })
}

/// Returns a copy of `input` where values from the height and width dimensions are moved to the depth dimension.
///
/// For example, given an input of shape `[1, 2, 2, 1]`, data_format = "NHWC" and
/// block_size = 2:
///
/// ```
/// x = [[[[1], [2]],
///       [[3], [4]]]]
/// ```
///
/// This operation will output a tensor of shape `[1, 1, 1, 4]`:
///
/// ```
/// [[[[1, 2, 3, 4]]]]
/// ```
///
/// Here, the input has a batch of 1 and each batch element has shape `[2, 2, 1]`,
/// the corresponding output will have a single element (i.e. width and height are
/// both 1) and will have a depth of 4 channels (1 * block_size * block_size).
/// The output element shape is `[1, 1, 4]`.
///
/// For an input tensor with larger depth, here of shape `[1, 2, 2, 3]`, e.g.
///
/// ```
/// x = [[[[1, 2, 3], [4, 5, 6]],
///       [[7, 8, 9], [10, 11, 12]]]]
/// ```
///
/// This operation, for block_size of 2, will return the following tensor of shape
/// `[1, 1, 1, 12]`
///
/// ```
/// [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
/// ```
///
/// Similarly, for the following input of shape `[1 4 4 1]`, and a block size of 2:
///
/// ```
/// x = [[[[1],   [2],  [5],  [6]],
///       [[3],   [4],  [7],  [8]],
///       [[9],  [10], [13],  [14]],
///       [[11], [12], [15],  [16]]]]
/// ```
///
/// the operator will return the following tensor of shape `[1 2 2 4]`:
///
/// ```
/// x = [[[[1, 2, 3, 4],
///        [5, 6, 7, 8]],
///       [[9, 10, 11, 12],
///        [13, 14, 15, 16]]]]
/// ```
///
/// - Precondition: `input.rank == 4 && b >= 2`.
/// - Precondition: The height of the input must be divisible by `b`.
/// - Precondition: The width of the input must be divisible by `b`.
@differentiable(reverse, wrt: input where Scalar: TensorFlowFloatingPoint)
public func spaceToDepth<Scalar>(_ input: Tensor<Scalar>, blockSize b: Int) -> Tensor<Scalar> {
  precondition(input.rank == 4, "The input must have rank 4.")
  precondition(b >= 2, "The block size must be greater than 1.")
  precondition(
    input.shape[1].isMultiple(of: b),
    "The height of the input must be divisible by the block size.")
  precondition(
    input.shape[2].isMultiple(of: b),
    "The width of the input must be divisible by the block size.")
  return _Raw.spaceToDepth(input, blockSize: Int64(b))
}

@usableFromInline
@derivative(of: spaceToDepth)
func _vjpSpaceToDepth<Scalar: TensorFlowFloatingPoint>(
  _ input: Tensor<Scalar>,
  blockSize b: Int
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> Tensor<Scalar>) {
  (spaceToDepth(input, blockSize: b), { depthToSpace($0, blockSize: b) })
}
