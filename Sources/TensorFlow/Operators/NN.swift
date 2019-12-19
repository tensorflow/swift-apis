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

//===------------------------------------------------------------------------------------------===//
// Normalization
//===------------------------------------------------------------------------------------------===//

public extension Tensor where Scalar: TensorFlowFloatingPoint {
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
    @differentiable(wrt: (self, offset, scale))
    func batchNormalized(
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

public extension Padding {
    @inlinable
    var raw: _Raw.Padding {
        switch self {
        case .same: return .same
        case .valid: return .valid
        }
    }

    @inlinable
    internal var raw2: _Raw.Padding2 {
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
@differentiable(wrt: (input, filter))
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
@differentiable(wrt: (input, filter), vjp: _vjpConv2D)
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
func _vjpConv2D<Scalar: TensorFlowFloatingPoint>(
    _ x: Tensor<Scalar>,
    filter: Tensor<Scalar>,
    strides: (Int, Int, Int, Int),
    padding: Padding,
    dilations: (Int, Int, Int, Int)
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
    let value = conv2D(x, filter: filter, strides: strides, padding: padding, dilations: dilations)
    return (value, { v in
        (conv2DBackpropInput(v, shape: x.shapeTensor, filter: filter,
                             strides: strides, padding: padding, dilations: dilations),
         conv2DBackpropFilter(v, input: x, filterSizes: filter.shapeTensor,
                              strides: strides, padding: padding, dilations: dilations))
    })
}

/// TensorFlow builtin conv2d gradient helper for the input.
@differentiable(wrt: (x, filter), vjp: _vjpConv2DBackpropInput)
@usableFromInline
func conv2DBackpropInput<Scalar: TensorFlowFloatingPoint>(
    _ x: Tensor<Scalar>,
    shape: Tensor<Int32>,
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

@usableFromInline
func _vjpConv2DBackpropInput<Scalar: TensorFlowFloatingPoint>(
    _ x: Tensor<Scalar>,
    _ shape: Tensor<Int32>,
    _ filter: Tensor<Scalar>,
    _ strides: (Int, Int, Int, Int),
    _ padding: Padding,
    _ dilations: (Int, Int, Int, Int)
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
    let value = conv2DBackpropInput(x, shape: shape, filter: filter,
                                    strides: strides, padding: padding, dilations: dilations)
    return (value, { v in
        (conv2D(v, filter: filter, strides: strides, padding: padding, dilations: dilations),
         conv2DBackpropFilter(x, input: v, filterSizes: filter.shapeTensor, strides: strides,
                              padding: padding, dilations: dilations))
    })
}

/// TensorFlow builtin conv2d gradient helper for the filter.
@differentiable(wrt: (x, input), vjp: _vjpConv2DBackpropFilter)
@usableFromInline
func conv2DBackpropFilter<Scalar: TensorFlowFloatingPoint>(
    _ x: Tensor<Scalar>,
    input: Tensor<Scalar>,
    filterSizes: Tensor<Int32>,
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
func _vjpConv2DBackpropFilter<Scalar: TensorFlowFloatingPoint>(
    _ x: Tensor<Scalar>,
    _ input: Tensor<Scalar>,
    _ filterSizes: Tensor<Int32>,
    _ strides: (Int, Int, Int, Int),
    _ padding: Padding,
    _ dilations: (Int, Int, Int, Int)
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
    let value = conv2DBackpropFilter(x, input: input, filterSizes: filterSizes,
                                     strides: strides, padding: padding, dilations: dilations)
    return (value, { v in
        (conv2D(input, filter: v, strides: strides, padding: padding, dilations: dilations),
         conv2DBackpropInput(x, shape: x.shapeTensor, filter: v, strides: strides,
                             padding: padding, dilations: dilations))
    })
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
@differentiable(wrt: (input, filter), vjp: _vjpConv3D)
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
        strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2),
                  Int32(strides.3), Int32(strides.4)],
        padding: padding.raw,
        dilations: [Int32(dilations.0), Int32(dilations.1), Int32(dilations.2),
                    Int32(dilations.3), Int32(dilations.4)]
    )
}

@usableFromInline
func _vjpConv3D<Scalar: TensorFlowFloatingPoint>(
    _ x: Tensor<Scalar>,
    filter: Tensor<Scalar>,
    strides: (Int, Int, Int, Int, Int),
    padding: Padding,
    dilations: (Int, Int, Int, Int, Int)
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
    let value = conv3D(x, filter: filter, strides: strides,
                       padding: padding, dilations: dilations)
    return (value, { v in
        (conv3DBackpropInput(v, shape: x.shapeTensor, filter: filter,
                             strides: strides, padding: padding),
         conv3DBackpropFilter(v, input: x, filterSizes: filter.shapeTensor,
                              strides: strides, padding: padding))
    })
}

/// TensorFlow builtin conv3d gradient helper for the input.
@differentiable(wrt: (x, filter), vjp: _vjpConv3DBackpropInput)
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
        strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2),
                  Int32(strides.3), Int32(strides.4)],
        padding: padding.raw,
        dilations: [Int32(dilations.0), Int32(dilations.1), Int32(dilations.2),
                    Int32(dilations.3), Int32(dilations.4)]
    )
}

@usableFromInline
func _vjpConv3DBackpropInput<Scalar: TensorFlowFloatingPoint>(
    _ x: Tensor<Scalar>,
    _ shape: Tensor<Int32>,
    _ filter: Tensor<Scalar>,
    _ strides: (Int, Int, Int, Int, Int),
    _ padding: Padding,
    _ dilations: (Int, Int, Int, Int, Int)
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
    let value = conv3DBackpropInput(x, shape: shape, filter: filter, strides: strides,
                                    padding: padding, dilations: dilations)
    return (value, { v in
        (conv3D(v, filter: filter, strides: strides, padding: padding),
         conv3DBackpropFilter(x, input: v, filterSizes: filter.shapeTensor, strides: strides,
                              padding: padding))
    })
}

/// TensorFlow builtin conv3d gradient helper for the filter.
@differentiable(wrt: (x, input), vjp: _vjpConv3DBackpropFilter)
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
        strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2),
                  Int32(strides.3), Int32(strides.4)],
        padding: padding.raw,
        dilations: [Int32(dilations.0), Int32(dilations.1), Int32(dilations.2),
                    Int32(dilations.3), Int32(dilations.4)]
    )
}

@usableFromInline
func _vjpConv3DBackpropFilter<Scalar: TensorFlowFloatingPoint>(
    _ x: Tensor<Scalar>,
    _ input: Tensor<Scalar>,
    _ filterSizes: Tensor<Int32>,
    _ strides: (Int, Int, Int, Int, Int),
    _ padding: Padding,
    _ dilations: (Int, Int, Int, Int, Int)
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
    let value = conv3DBackpropFilter(x, input: input, filterSizes: filterSizes,
                                     strides: strides, padding: padding, dilations: dilations)
    return (value, { v in
        (conv3D(input, filter: v, strides: strides, padding: padding),
         conv3DBackpropInput(x, shape: x.shapeTensor, filter: v, strides: strides,
                             padding: padding))
    })
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
@differentiable(wrt: (input, filter), vjp: _vjpDepthwiseConv2D)
public func depthwiseConv2D<Scalar: TensorFlowFloatingPoint>(
    _ input: Tensor<Scalar>,
    filter: Tensor<Scalar>,
    strides: (Int, Int, Int, Int),
    padding: Padding
) -> Tensor<Scalar> {
    precondition(input.shape.rank == 4, "The input must have rank 4.")
    precondition(filter.shape.rank == 4, "The filter must have rank 4.")
    return _Raw.depthwiseConv2dNative(
        input,
        filter: filter,
        strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2),Int32(strides.3)],
        padding: padding.raw)
}

@usableFromInline
func _vjpDepthwiseConv2D<Scalar: TensorFlowFloatingPoint>(
    _ x: Tensor<Scalar>,
    filter: Tensor<Scalar>,
    strides: (Int, Int, Int, Int),
    padding: Padding
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
    let value = depthwiseConv2D(x, filter: filter, strides: strides,
                                padding: padding)
    return (value, { v in
        (depthwiseConv2dBackpropInput(v, shape: x.shapeTensor, filter: filter,
                                      strides: strides, padding: padding),
         depthwiseConv2dBackpropFilter(v, input: x, filterSizes: filter.shapeTensor,
                                       strides: strides, padding: padding))
    })
}

/// TensorFlow builtin depthwiseConv2D gradient helper for the input.
@differentiable(wrt: (x, filter), vjp: _vjpDepthwiseConv2dBackpropInput)
@usableFromInline
func depthwiseConv2dBackpropInput<Scalar: TensorFlowFloatingPoint>(
    _ x: Tensor<Scalar>,
    shape: Tensor<Int32>,
    filter: Tensor<Scalar>,
    strides: (Int, Int, Int, Int),
    padding: Padding
) -> Tensor<Scalar> {
    return _Raw.depthwiseConv2dNativeBackpropInput(
        inputSizes: shape,
        filter: filter,
        outBackprop: x,
        strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3)],
        padding: padding.raw)
}

@usableFromInline
func _vjpDepthwiseConv2dBackpropInput<Scalar: TensorFlowFloatingPoint>(
    _ x: Tensor<Scalar>,
    _ shape: Tensor<Int32>,
    _ filter: Tensor<Scalar>,
    _ strides: (Int, Int, Int, Int),
    _ padding: Padding
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
    let value = depthwiseConv2dBackpropInput(x, shape: shape, filter: filter, strides: strides,
                                             padding: padding)
    return (value, { v in
        (depthwiseConv2D(v, filter: filter, strides: strides, padding: padding),
         depthwiseConv2dBackpropFilter(x, input: v, filterSizes: filter.shapeTensor,
                                       strides: strides, padding: padding))

    })
}

/// TensorFlow builtin depthwiseConv2D gradient helper for the filter.
@differentiable(wrt: (x, input), vjp: _vjpDepthwiseConv2dBackpropFilter)
@usableFromInline
func depthwiseConv2dBackpropFilter<Scalar: TensorFlowFloatingPoint>(
    _ x: Tensor<Scalar>,
    input: Tensor<Scalar>,
    filterSizes: Tensor<Int32>,
    strides: (Int, Int, Int, Int),
    padding: Padding
) -> Tensor<Scalar> {
    return _Raw.depthwiseConv2dNativeBackpropFilter(
        input,
        filterSizes: filterSizes,
        outBackprop: x,
        strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3)],
        padding: padding.raw)
}

@usableFromInline
func _vjpDepthwiseConv2dBackpropFilter<Scalar: TensorFlowFloatingPoint>(
    _ x: Tensor<Scalar>,
    _ input: Tensor<Scalar>,
    _ filterSizes: Tensor<Int32>,
    _ strides: (Int, Int, Int, Int),
    _ padding: Padding
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
    let value = depthwiseConv2dBackpropFilter(x, input: input, filterSizes: filterSizes,
                                              strides: strides, padding: padding)
    return (value, { v in
        (depthwiseConv2D(input, filter: v, strides: strides, padding: padding),
         depthwiseConv2dBackpropInput(x, shape: x.shapeTensor, filter: v, strides: strides,
                                      padding: padding))
    })
}

/// Returns a 2-D max pooling, with the specified filter sizes, strides, and
/// padding.
///
/// - Parameters:
///   - input: The input.
///   - filterSize: The dimensions of the pooling kernel.
///   - strides: The strides of the sliding filter for each dimension of the input.
///   - padding: The padding for the operation.
@differentiable(wrt: input, vjp: _vjpMaxPool2D)
public func maxPool2D<Scalar: TensorFlowFloatingPoint>(
    _ input: Tensor<Scalar>,
    filterSize: (Int, Int, Int, Int),
    strides: (Int, Int, Int, Int),
    padding: Padding
) -> Tensor<Scalar> {
    return _Raw.maxPoolV2(
        input,
        ksize: Tensor<Int32>([Int32(filterSize.0), Int32(filterSize.1),
                                   Int32(filterSize.2), Int32(filterSize.3)]),
        strides: Tensor<Int32>([Int32(strides.0), Int32(strides.1),
                                Int32(strides.2), Int32(strides.3)]),
        padding: padding.raw)
}

@usableFromInline
func _vjpMaxPool2D<Scalar: TensorFlowFloatingPoint>(
    _ x: Tensor<Scalar>,
    filterSize: (Int, Int, Int, Int),
    strides: (Int, Int, Int, Int),
    padding: Padding
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> Tensor<Scalar>) {
    // TODO: Currently this is not higher order differentiable. Redefine in
    // closed form.
    let value = maxPool2D(x, filterSize: filterSize, strides: strides, padding: padding)
    return (value, { v in
        _Raw.maxPoolGradV2(
            origInput: x,
            origOutput: value,
            grad: v,
            ksize: Tensor<Int32>([Int32(filterSize.0), Int32(filterSize.1),
                                  Int32(filterSize.2), Int32(filterSize.3)]),
            strides: Tensor<Int32>([Int32(strides.0), Int32(strides.1),
                                    Int32(strides.2), Int32(strides.3)]),
            padding: padding.raw)
    })
}

/// Returns a 3-D max pooling, with the specified filter sizes, strides, and
/// padding.
///
/// - Parameters:
///   - input: The input.
///   - filterSize: The dimensions of the pooling kernel.
///   - strides: The strides of the sliding filter for each dimension of the input.
///   - padding: The padding for the operation.
@differentiable(wrt: input, vjp: _vjpMaxPool3D)
public func maxPool3D<Scalar: TensorFlowFloatingPoint>(
    _ input: Tensor<Scalar>,
    filterSize: (Int, Int, Int, Int, Int),
    strides: (Int, Int, Int, Int, Int),
    padding: Padding
) -> Tensor<Scalar> {
    return _Raw.maxPool3D(
        input,
        ksize: [Int32(filterSize.0), Int32(filterSize.1),
                     Int32(filterSize.2), Int32(filterSize.3), Int32(filterSize.4)],
        strides: [Int32(strides.0), Int32(strides.1),
                  Int32(strides.2), Int32(strides.3), Int32(strides.4)],
        padding: padding.raw)
}

@usableFromInline
func _vjpMaxPool3D<Scalar: TensorFlowFloatingPoint>(
    _ x: Tensor<Scalar>,
    filterSize: (Int, Int, Int, Int, Int),
    strides: (Int, Int, Int, Int, Int),
    padding: Padding
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> Tensor<Scalar>) {
    // TODO: Currently this is not higher order differentiable. Redefine in
    // closed form.
    let value = maxPool3D(x, filterSize: filterSize, strides: strides, padding: padding)
    return (value, { v in
        return _Raw.maxPool3DGrad(
            origInput: x,
            origOutput: value,
            grad: v,
            ksize: [Int32(filterSize.0), Int32(filterSize.1), Int32(filterSize.2),
                    Int32(filterSize.3), Int32(filterSize.4)],
            strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3),
                      Int32(strides.4)],
            padding: padding.raw
        )
    })
}

/// Returns a 2-D average pooling, with the specified filter sizes, strides,
/// and padding.
///
/// - Parameters:
///   - input: The input.
///   - filterSize: The dimensions of the pooling kernel.
///   - strides: The strides of the sliding filter for each dimension of the input.
///   - padding: The padding for the operation.
@differentiable(wrt: input, vjp: _vjpAvgPool2D)
public func avgPool2D<Scalar: TensorFlowFloatingPoint>(
    _ input: Tensor<Scalar>,
    filterSize: (Int, Int, Int, Int),
    strides: (Int, Int, Int, Int),
    padding: Padding
) -> Tensor<Scalar> {
    return _Raw.avgPool(
        value: input,
        ksize: [Int32(filterSize.0), Int32(filterSize.1),
                Int32(filterSize.2), Int32(filterSize.3)],
        strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3)],
        padding: padding.raw)
}

@usableFromInline
func _vjpAvgPool2D<Scalar: TensorFlowFloatingPoint>(
    _ x: Tensor<Scalar>,
    filterSize: (Int, Int, Int, Int),
    strides: (Int, Int, Int, Int),
    padding: Padding
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> Tensor<Scalar>) {
    // TODO: Currently this is not higher order differentiable. Redefine in
    // closed form.
    let value = avgPool2D(x, filterSize: filterSize, strides: strides, padding: padding)
    return (value, { v in
        _Raw.avgPoolGrad(
            origInputShape: x.shapeTensor,
            grad: v,
            ksize: [Int32(filterSize.0), Int32(filterSize.1),
                    Int32(filterSize.2), Int32(filterSize.3)],
            strides: [Int32(strides.0), Int32(strides.1),
                      Int32(strides.2), Int32(strides.3)],
            padding: padding.raw
        )
    })
}

/// Returns a 3-D average pooling, with the specified filter sizes, strides,
/// and padding.
///
/// - Parameters:
///   - input: The input.
///   - filterSize: The dimensions of the pooling kernel.
///   - strides: The strides of the sliding filter for each dimension of the input.
///   - padding: The padding for the operation.
@differentiable(wrt: input, vjp: _vjpAvgPool3D)
public func avgPool3D<Scalar: TensorFlowFloatingPoint>(
    _ input: Tensor<Scalar>,
    filterSize: (Int, Int, Int, Int, Int),
    strides: (Int, Int, Int, Int, Int),
    padding: Padding
) -> Tensor<Scalar> {
    return _Raw.avgPool3D(
        input,
        ksize: [Int32(filterSize.0), Int32(filterSize.1),
                Int32(filterSize.2), Int32(filterSize.3), Int32(filterSize.4)],
        strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3),
                  Int32(strides.4)],
        padding: padding.raw)
}

@usableFromInline
func _vjpAvgPool3D<Scalar: TensorFlowFloatingPoint>(
    _ x: Tensor<Scalar>,
    filterSize: (Int, Int, Int, Int, Int),
    strides: (Int, Int, Int, Int, Int),
    padding: Padding
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> Tensor<Scalar>) {
    // TODO: Currently this is not higher order differentiable. Redefine in
    // closed form.
    let value = avgPool3D(x, filterSize: filterSize, strides: strides, padding: padding)
    return (value, { v in
        return _Raw.avgPool3DGrad(
            origInputShape: x.shapeTensor,
            grad: v,
            ksize: [Int32(filterSize.0), Int32(filterSize.1), Int32(filterSize.2),
                    Int32(filterSize.3), Int32(filterSize.4)],
            strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3),
                      Int32(strides.4)],
            padding: padding.raw
        )
    })
}
