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
    @differentiable(wrt: (self, offset, scale), vjp: _vjpBatchNormalized)
    func batchNormalized(
        alongAxis axis: Int,
        offset: Tensor = Tensor(0),
        scale: Tensor = Tensor(1),
        epsilon: Scalar = 0.001
    ) -> Tensor {
        let mean = self.mean(alongAxes: axis)
        let squaredDiff: Tensor = Raw.squaredDifference(self, mean)
        let variance = squaredDiff.mean(alongAxes: axis)
        let inv = rsqrt(variance + epsilon) * scale
        return self * inv + offset - mean * inv
    }

    // TODO: Verify that these calculations are correct.
    @inlinable
    internal func _vjpBatchNormalized(
        alongAxis axis: Int,
        offset: Tensor,
        scale: Tensor,
        epsilon: Scalar
    ) -> (Tensor, (Tensor) -> (Tensor, Tensor, Tensor)) {
        let value = batchNormalized(alongAxis: axis, offset: offset, scale: scale, epsilon: epsilon)
        return (value, { v in
            let mean = self.mean(alongAxes: axis)
            let squaredDiff: Tensor = Raw.squaredDifference(self, mean)
            let variance = squaredDiff.mean(alongAxes: axis)

            let diff = self - mean
            let inv = rsqrt(variance + epsilon)
            let norm = diff * inv

            let dNorm = v * scale
            let dVariance = -(dNorm * diff).sum(alongAxes: axis) / 2 * TensorFlow.pow(inv, -3)
            // Note: `dMean` is split into two lines to avoid the "compiler is unable to type-check
            // this expression in reasonable time" error.
            var dMean = (-dNorm * inv).sum(alongAxes: axis)
            dMean = dMean + dVariance * (-diff * 2).mean(alongAxes: axis)
            let dOffset = v.sum(alongAxes: axis)
            let dScale = (norm * v).sum(alongAxes: axis)
            let dim = Tensor(Tensor<Int32>(self.shapeTensor[axis]))
            let tmp = (dNorm * inv) + (dVariance * 2 * dMean / dim)
            let dSelf = tmp + (dMean / dim)
            return (dSelf, dOffset, dScale)
        })
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
    var raw: Raw.Padding {
        switch self {
        case .same: return .same
        case .valid: return .valid
        }
    }

    @inlinable
    internal var raw2: Raw.Padding2 {
        switch self {
        case .same: return .same
        case .valid: return .valid
        }
    }
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
    return Raw.conv2D(
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
    return Raw.conv2DBackpropInput(
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
    return Raw.conv2DBackpropFilter(
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
        (conv2DBackpropInput(x, shape: filterSizes, filter: v, strides: strides,
                             padding: padding, dilations: dilations),
         conv2D(input, filter: v, strides: strides, padding: padding, dilations: dilations))
    })
}

/// Returns a 3-D convolution with the specified input, filter, strides, and padding.
///
/// - Parameters:
///   - input: The input.
///   - filter: The convolution filter.
///   - strides: The strides of the sliding filter for each dimension of the input.
///   - padding: The padding for the operation.
/// - Precondition: `input` must have rank `5`.
/// - Precondition: `filter` must have rank 5.
@differentiable(wrt: (input, filter), vjp: _vjpConv3D)
public func conv3D<Scalar: TensorFlowFloatingPoint>(
    _ input: Tensor<Scalar>,
    filter: Tensor<Scalar>,
    strides: (Int, Int, Int, Int, Int) = (1, 1, 1, 1, 1),
    padding: Padding = .valid
) -> Tensor<Scalar> {
    return Raw.conv3D(
        input,
        filter: filter,
        strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2),
                  Int32(strides.3), Int32(strides.4)],
        padding: padding.raw)
}

@usableFromInline
func _vjpConv3D<Scalar: TensorFlowFloatingPoint>(
    _ x: Tensor<Scalar>,
    filter: Tensor<Scalar>,
    strides: (Int, Int, Int, Int, Int),
    padding: Padding
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
    let value = conv3D(x, filter: filter, strides: strides,
                       padding: padding)
    return (value, { v in
        return (
            conv3DBackpropInput(v, shape: x.shapeTensor, filter: filter,
                                strides: strides, padding: padding),
            conv3DBackpropFilter(v, input: x, filterSizes: filter.shapeTensor,
                                 strides: strides, padding: padding)
        )
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
    padding: Padding = .valid
) -> Tensor<Scalar> {
    return Raw.conv3DBackpropInputV2(
        inputSizes: shape,
        filter: filter,
        outBackprop: x,
        strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2),
                  Int32(strides.3), Int32(strides.4)],
        padding: padding.raw)
}

@usableFromInline
func _vjpConv3DBackpropInput<Scalar: TensorFlowFloatingPoint>(
    _ x: Tensor<Scalar>,
    _ shape: Tensor<Int32>,
    _ filter: Tensor<Scalar>,
    _ strides: (Int, Int, Int, Int, Int),
    _ padding: Padding
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
    let value = conv3DBackpropInput(x, shape: shape, filter: filter, strides: strides,
                                    padding: padding)
    return (value, { v in
        return (
            conv3DBackpropFilter(x, input: v, filterSizes: shape, strides: strides,
                                 padding: padding),
            conv3D(v, filter: filter, strides: strides, padding: padding)
        )
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
    padding: Padding = .valid
) -> Tensor<Scalar> {
    return Raw.conv3DBackpropFilterV2(
        x,
        filterSizes: filterSizes,
        outBackprop: x,
        strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2),
                  Int32(strides.3), Int32(strides.4)],
        padding: padding.raw)
}

@usableFromInline
func _vjpConv3DBackpropFilter<Scalar: TensorFlowFloatingPoint>(
    _ x: Tensor<Scalar>,
    _ input: Tensor<Scalar>,
    _ filterSizes: Tensor<Int32>,
    _ strides: (Int, Int, Int, Int, Int),
    _ padding: Padding
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
    let value = conv3DBackpropFilter(x, input: input, filterSizes: filterSizes,
                                     strides: strides, padding: padding)
    return (value, { v in
        return (
            conv3DBackpropInput(x, shape: filterSizes, filter: v, strides: strides,
                                  padding: padding),
            conv3D(input, filter: v, strides: strides, padding: padding)
        )
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
    return Raw.depthwiseConv2dNative(
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
        return (
            depthwiseConv2dBackpropInput(v, shape: x.shapeTensor, filter: filter,
                                         strides: strides, padding: padding
            ),
            depthwiseConv2dBackpropFilter(v, input: x, filterSizes: filter.shapeTensor,
                                          strides: strides, padding: padding
            )
        )
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
    return Raw.depthwiseConv2dNativeBackpropInput(
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
        return (
            depthwiseConv2dBackpropFilter(x, input: v, filterSizes: shape, strides: strides,
                                          padding: padding),
            depthwiseConv2D(v, filter: filter, strides: strides, padding: padding)
        )
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
    return Raw.depthwiseConv2dNativeBackpropFilter(
        x,
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
        return (
            depthwiseConv2dBackpropInput(x, shape: filterSizes, filter: v, strides: strides,
                                         padding: padding),
            depthwiseConv2D(input, filter: v, strides: strides, padding: padding)
        )
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
    return Raw.maxPoolV2(
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
        Raw.maxPoolGradV2(
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
    return Raw.maxPool3D(
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
        return Raw.maxPool3DGrad(
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
    return Raw.avgPool(
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
        Raw.avgPoolGrad(
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
    return Raw.avgPool3D(
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
        return Raw.avgPool3DGrad(
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
