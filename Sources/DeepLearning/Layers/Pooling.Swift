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

/// An average pooling layer for temporal data.
@_fixed_layout
public struct AvgPool1D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The size of the sliding reduction window for pooling.
    @noDerivative let poolSize: Int
    /// The stride of the sliding window for temporal dimension.
    @noDerivative let stride: Int
    /// The padding algorithm for pooling.
    @noDerivative let padding: Padding

    /// Creates an average pooling layer.
    ///
    /// - Parameters:
    ///   - poolSize: The size of the sliding reduction window for pooling.
    ///   - stride: The stride of the sliding window for temporal dimension.
    ///   - padding: The padding algorithm for pooling.
    public init(
        poolSize: Int,
        stride: Int,
        padding: Padding
    ) {
        self.poolSize = poolSize
        self.stride = stride
        self.padding = padding
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func call(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.expandingShape(at: 1).averagePooled(
            kernelSize: (1, 1, poolSize, 1), strides: (1, 1, stride, 1), padding: padding
        ).squeezingShape(at: 1)
    }
}

/// An average pooling layer for spatial data.
@_fixed_layout
public struct AvgPool2D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The size of the sliding reduction window for pooling.
    @noDerivative let poolSize: (Int, Int, Int, Int)
    /// The strides of the sliding window for each dimension of a 4-D input.
    /// Strides in non-spatial dimensions must be `1`.
    @noDerivative let strides: (Int, Int, Int, Int)
    /// The padding algorithm for pooling.
    @noDerivative let padding: Padding

    /// Creates a average pooling layer.
    public init(
        poolSize: (Int, Int, Int, Int),
        strides: (Int, Int, Int, Int),
        padding: Padding
    ) {
        self.poolSize = poolSize
        self.strides = strides
        self.padding = padding
    }

    /// Creates a average pooling layer.
    ///
    /// - Parameters:
    ///   - poolSize: Vertical and horizontal factors by which to downscale.
    ///   - strides: The strides.
    ///   - padding: The padding.
    public init(poolSize: (Int, Int), strides: (Int, Int), padding: Padding = .valid) {
        self.poolSize = (1, poolSize.0, poolSize.1, 1)
        self.strides = (1, strides.0, strides.1, 1)
        self.padding = padding
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func call(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.averagePooled(kernelSize: poolSize, strides: strides, padding: padding)
    }
}

/// A global average pooling layer for temporal data.
@_fixed_layout
public struct GlobalAvgPool1D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// Creates a global average pooling layer.
    public init() {}

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func call(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.mean(squeezingAxes: 1)
    }
}

/// A global average pooling layer for spatial data.
@_fixed_layout
public struct GlobalAvgPool2D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// Creates a global average pooling layer.
    public init() {}

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func call(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.mean(squeezingAxes: [1, 2])
    }
}

/// A global average pooling layer for spatial and spatio-temporal data.
@_fixed_layout
public struct GlobalAvgPool3D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// Creates a global average pooling layer.
    public init() {}

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func call(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.mean(squeezingAxes: [1, 2, 3])
    }
}

/// A max pooling layer for temporal data.
@_fixed_layout
public struct MaxPool1D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The size of the sliding reduction window for pooling.
    @noDerivative let poolSize: Int
    /// The stride of the sliding window for temporal dimension.
    @noDerivative let stride: Int
    /// The padding algorithm for pooling.
    @noDerivative let padding: Padding

    /// Creates a max pooling layer.
    ///
    /// - Parameters:
    ///   - poolSize: The size of the sliding reduction window for pooling.
    ///   - stride: The stride of the sliding window for temporal dimension.
    ///   - padding: The padding algorithm for pooling.
    public init(
        poolSize: Int,
        stride: Int,
        padding: Padding
    ) {
        self.poolSize = poolSize
        self.stride = stride
        self.padding = padding
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func call(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.expandingShape(at: 1).maxPooled(
            kernelSize: (1, 1, poolSize, 1), strides: (1, 1, stride, 1), padding: padding
        ).squeezingShape(at: 1)
    }
}

/// A max pooling layer for spatial data.
@_fixed_layout
public struct MaxPool2D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The size of the sliding reduction window for pooling.
    @noDerivative let poolSize: (Int, Int, Int, Int)
    /// The strides of the sliding window for each dimension of a 4-D input.
    /// Strides in non-spatial dimensions must be `1`.
    @noDerivative let strides: (Int, Int, Int, Int)
    /// The padding algorithm for pooling.
    @noDerivative let padding: Padding

    /// Creates a max pooling layer.
    public init(
        poolSize: (Int, Int, Int, Int),
        strides: (Int, Int, Int, Int),
        padding: Padding
    ) {
        self.poolSize = poolSize
        self.strides = strides
        self.padding = padding
    }

    /// Creates a max pooling layer.
    ///
    /// - Parameters:
    ///   - poolSize: Vertical and horizontal factors by which to downscale.
    ///   - strides: The strides.
    ///   - padding: The padding.
    public init(poolSize: (Int, Int), strides: (Int, Int), padding: Padding = .valid) {
        self.poolSize = (1, poolSize.0, poolSize.1, 1)
        self.strides = (1, strides.0, strides.1, 1)
        self.padding = padding
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func call(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.maxPooled(
            kernelSize: poolSize, strides: strides, padding: padding)
    }
}
