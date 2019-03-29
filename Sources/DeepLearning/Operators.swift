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

/// Returns the values of the specified tensor rounded to the nearest integer, element-wise.
public func round<Scalar: BinaryFloatingPoint>(_ x: Tensor<Scalar>) -> Tensor<Scalar> {
    return Raw.round(x)
}

/// Returns a tensor with the same shape and scalars as the specified tensor.
@differentiable
public func identity<Scalar>(_ x: Tensor<Scalar>) -> Tensor<Scalar> {
    return x
}

//===------------------------------------------------------------------------------------------===//
// Normalization
//===------------------------------------------------------------------------------------------===//

public extension Tensor where Scalar: TensorFlowFloatingPoint {
    // TODO: Verify that these calculations are correct.
    @inlinable
    internal func _vjpBatchNormalized(
        alongAxis axis: Int32,
        offset: Tensor,
        scale: Tensor,
        epsilon: Scalar
    ) -> (Tensor, (Tensor) -> (Tensor, Tensor, Tensor)) {
        let value = batchNormalized(alongAxis: axis, offset: offset, scale: scale,
                                    epsilon: epsilon)
        return (value, { v in
            let mean = self.mean(alongAxes: axis)
            let squaredDiff: Tensor = Raw.squaredDifference(self, mean)
            let variance = squaredDiff.mean(alongAxes: axis)

             let diff = self - mean
            let inv = rsqrt(variance + epsilon)
            let norm = diff * inv

             let dNorm = v * scale
            let dVariance = -(dNorm * diff).sum(alongAxes: axis) / 2 * pow(inv, -3)
            let dMean = (-dNorm * inv).sum(alongAxes: axis) +
                dVariance * (-diff * 2).mean(alongAxes: axis)
            let dOffset = v.sum(alongAxes: axis)
            let dScale = (norm * v).sum(alongAxes: axis)
            let dim = Tensor(Tensor<Int32>(self.shapeTensor[axis]))
            let tmp = (dNorm * inv) + (dVariance * 2 * dMean / dim)
            let dSelf = tmp + (dMean / dim)
            return (dSelf, dOffset, dScale)
        })
    }
}

public extension Tensor where Scalar: BinaryFloatingPoint {
    /// Computes the batch normalized tensor along the specified axis.
    ///
    /// Specifically, returns `(self - mu)/(var + epsilon) * gamma + beta` where
    /// `mu` and `var` are respectively the mean and variance of `self` along
    /// `axis`.
    ///
    /// - Parameters:
    ///     - axis: The batch dimension.
    ///     - offset: The offset, also known as beta.
    ///     - scale: The scale, also known as gamma.
    ///     - epsilon: A small value added to the denominator for numerical
    ///         stability.
    @inlinable
    @differentiable(
        wrt: (self, offset, scale), vjp: _vjpBatchNormalized
        where Scalar : TensorFlowFloatingPoint
    )
    func batchNormalized(
        alongAxis axis: Int32,
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
}

//===------------------------------------------------------------------------------------------===//
// Convolution and pooling
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
}

public extension Tensor where Scalar: TensorFlowFloatingPoint {
    /// TensorFlow builtin conv2d gradient helper for the input.
    @inlinable
    @differentiable(wrt: (self, filter), vjp: _vjpConv2DBackpropInput)
    internal func conv2DBackpropInput(
        shape: Tensor<Int32>,
        filter: Tensor,
        strides: (Int32, Int32, Int32, Int32),
        padding: Padding
    ) -> Tensor {
        return Raw.conv2DBackpropInput(
            inputSizes: shape,
            filter: filter,
            outBackprop: self,
            strides: [strides.0, strides.1, strides.2, strides.3],
            padding: padding.raw)
    }

    /// TensorFlow builtin conv2d gradient helper for the filter.
    @inlinable
    @differentiable(wrt: (self, input), vjp: _vjpConv2DBackpropFilter)
    internal func conv2DBackpropFilter(
        input: Tensor,
        filterSizes: Tensor<Int32>,
        strides: (Int32, Int32, Int32, Int32),
        padding: Padding
    ) -> Tensor {
        return Raw.conv2DBackpropFilter(
            input,
            filterSizes: filterSizes,
            outBackprop: self,
            strides: [strides.0, strides.1, strides.2, strides.3],
            padding: padding.raw)
    }

    @inlinable
    internal func _vjpConv2DBackpropInput(
        _ shape: Tensor<Int32>,
        _ filter: Tensor,
        _ strides: (Int32, Int32, Int32, Int32),
        _ padding: Padding
    ) -> (Tensor, (Tensor) -> (Tensor, Tensor)) {
        let value = conv2DBackpropInput(shape: shape, filter: filter, strides: strides,
                                        padding: padding)
        return (value, { v in
            return (
                self.conv2DBackpropFilter(input: v, filterSizes: shape, strides: strides,
                                          padding: padding),
                v.convolved2D(withFilter: filter, strides: strides, padding: padding)
            )
        })
    }

    @inlinable
    internal func _vjpConv2DBackpropFilter(
        _ input: Tensor,
        _ filterSizes: Tensor<Int32>,
        _ strides: (Int32, Int32, Int32, Int32),
        _ padding: Padding
    ) -> (Tensor, (Tensor) -> (Tensor, Tensor)) {
        let value = conv2DBackpropFilter(input: input, filterSizes: filterSizes,
                                         strides: strides, padding: padding)
        return (value, { v in
            return (
                self.conv2DBackpropInput(shape: filterSizes, filter: v, strides: strides,
                                         padding: padding),
                input.convolved2D(withFilter: v, strides: strides, padding: padding)
            )
        })
    }

    @inlinable
    internal func _vjpConvolved2D(
        filter: Tensor,
        strides: (Int32, Int32, Int32, Int32),
        padding: Padding
    ) -> (Tensor, (Tensor) -> (Tensor, Tensor)) {
        let value = convolved2D(withFilter: filter, strides: strides,
                                padding: padding)
        return (value, { v in
            return (
                v.conv2DBackpropInput(
                    shape: self.shapeTensor, filter: filter,
                    strides: strides, padding: padding
                ),
                v.conv2DBackpropFilter(
                    input: self, filterSizes: filter.shapeTensor,
                    strides: strides, padding: padding
                )
            )
        })
    }

    @inlinable
    internal func _vjpMaxPooled(
        kernelSize: (Int32, Int32, Int32, Int32),
        strides: (Int32, Int32, Int32, Int32),
        padding: Padding
    ) -> (Tensor, (Tensor) -> Tensor) {
        // TODO: Currently this is not higher order differentiable. Redefine in
        // closed form.
        let value = maxPooled(kernelSize: kernelSize, strides: strides,
                              padding: padding)
        return (value, { v in
            return Raw.maxPoolGradV2(
                origInput: self,
                origOutput: value,
                grad: v,
                ksize: Tensor<Int32>([kernelSize.0, kernelSize.1, kernelSize.2, kernelSize.3]),
                strides: Tensor<Int32>([strides.0, strides.1, strides.2, strides.3]),
                padding: padding.raw
            )
        })
    }

    @inlinable
    internal func _vjpAveragePooled(
        kernelSize: (Int32, Int32, Int32, Int32),
        strides: (Int32, Int32, Int32, Int32),
        padding: Padding
    ) -> (Tensor, (Tensor) -> Tensor) {
        // TODO: Currently this is not higher order differentiable. Redefine in
        // closed form.
        let value = averagePooled(kernelSize: kernelSize, strides: strides,
                                  padding: padding)
        return (value, { v in
            return Raw.avgPoolGrad(
                origInputShape: self.shapeTensor,
                grad: v,
                ksize: [kernelSize.0, kernelSize.1, kernelSize.2, kernelSize.3],
                strides: [strides.0, strides.1, strides.2, strides.3],
                padding: padding.raw
            )
        })
    }
}

public extension Tensor where Scalar: FloatingPoint {
    /// Computes a 2-D convolution using `self` as input, with the specified
    /// filter, strides, and padding.
    ///
    /// - Parameters:
    ///     - filter: The convolution filter.
    ///     - strides: The strides of the sliding filter for each dimension of the
    ///         input.
    ///     - padding: The padding for the operation.
    /// - Precondition: `self` must have rank 4.
    /// - Precondition: `filter` must have rank 4.
    @inlinable @inline(__always)
    @differentiable(
        wrt: (self, filter), vjp: _vjpConvolved2D
        where Scalar: TensorFlowFloatingPoint
    )
    func convolved2D(
        withFilter filter: Tensor,
        strides: (Int32, Int32, Int32, Int32),
        padding: Padding
    ) -> Tensor {
        return Raw.conv2D(
            self,
            filter: filter,
            strides: [strides.0, strides.1, strides.2, strides.3],
            padding: padding.raw)
    }

    /// Computes a 2-D max pooling, with the specified kernel sizes, strides, and
    /// padding.
    ///
    /// - Parameters:
    ///     - kernelSize: The dimensions of the pooling kernel.
    ///     - strides: The strides of the sliding filter for each dimension of the
    ///         input.
    ///     - padding: The padding for the operation.
    @inlinable @inline(__always)
    @differentiable(
        wrt: self, vjp: _vjpMaxPooled(kernelSize:strides:padding:)
        where Scalar : TensorFlowFloatingPoint
    )
    func maxPooled(
        kernelSize: (Int32, Int32, Int32, Int32),
        strides: (Int32, Int32, Int32, Int32),
        padding: Padding
    ) -> Tensor {
        return Raw.maxPoolV2(
            self,
            ksize: Tensor<Int32>([kernelSize.0, kernelSize.1,
                                 kernelSize.2, kernelSize.3]),
            strides: Tensor<Int32>([strides.0, strides.1,
                                   strides.2, strides.3]),
            padding: padding.raw)
    }

    /// Computes a 2-D average pooling, with the specified kernel sizes, strides,
    /// and padding.
    ///
    /// - Parameters:
    ///     - kernelSize: The dimensions of the pooling kernel.
    ///     - strides: The strides of the sliding filter for each dimension of the
    ///         input.
    ///     - padding: The padding for the operation.
    @inlinable @inline(__always)
    @differentiable(
        wrt: self, vjp: _vjpAveragePooled(kernelSize:strides:padding:)
        where Scalar : TensorFlowFloatingPoint
    )
    func averagePooled(
        kernelSize: (Int32, Int32, Int32, Int32),
        strides: (Int32, Int32, Int32, Int32),
        padding: Padding
    ) -> Tensor {
        return Raw.avgPool(
            value: self,
            ksize: [kernelSize.0, kernelSize.1, kernelSize.2, kernelSize.3],
            strides: [strides.0, strides.1, strides.2, strides.3],
            padding: padding.raw)
    }
}
