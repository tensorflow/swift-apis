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
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func call(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.expandingShape(at: 1).maxPooled2D(
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

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func call(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.maxPooled2D(
            kernelSize: poolSize, strides: strides, padding: padding)
    }
}

public extension MaxPool2D {
  /// Creates a max pooling layer.
  ///
  /// - Parameters:
  ///   - poolSize: Vertical and horizontal factors by which to downscale.
  ///   - strides: The strides.
  ///   - padding: The padding.
  init(poolSize: (Int, Int), strides: (Int, Int), padding: Padding = .valid) {
  self.init(poolSize: (1, poolSize.0, poolSize.1, 1),
            strides: (1, strides.0, strides.1, 1),
            padding: padding)
  }
}

/// A max pooling layer for spatial or spatio-temporal data.
@_fixed_layout
public struct MaxPool3D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The size of the sliding reduction window for pooling.
    @noDerivative let poolSize: (Int, Int, Int, Int, Int)
    /// The strides of the sliding window for each dimension of a 5-D input.
    /// Strides in non-spatial dimensions must be `1`.
    @noDerivative let strides: (Int, Int, Int, Int, Int)
    /// The padding algorithm for pooling.
    @noDerivative let padding: Padding

    /// Creates a max pooling layer.
    public init(
        poolSize: (Int, Int, Int, Int, Int),
        strides: (Int, Int, Int, Int, Int),
        padding: Padding
    ) {
        self.poolSize = poolSize
        self.strides = strides
        self.padding = padding
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func call(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.maxPooled3D(kernelSize: poolSize, strides: strides, padding: padding)
    }
}

public extension MaxPool3D {
    /// Creates a max pooling layer.
    ///
    /// - Parameters:
    ///   - poolSize: Vertical and horizontal factors by which to downscale.
    ///   - strides: The strides.
    ///   - padding: The padding.
    init(poolSize: (Int, Int, Int), strides: (Int, Int, Int), padding: Padding = .valid) {
        self.init(poolSize: (1, poolSize.0, poolSize.1, poolSize.2, 1),
                  strides: (1, strides.0, strides.1, strides.2, 1),
                  padding: padding)
  }
}

public extension MaxPool3D {
    /// Creates a max pooling layer with the specified pooling window size and stride. All
    /// pooling sizes and strides are the same.
    init(poolSize: Int, stride: Int, padding: Padding = .valid) {
        self.init(poolSize: (poolSize, poolSize, poolSize),
                  strides: (stride, stride, stride),
                  padding: padding)
  }
}

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
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func call(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.expandingShape(at: 1).averagePooled2D(
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

    /// Creates an average pooling layer.
    public init(
        poolSize: (Int, Int, Int, Int),
        strides: (Int, Int, Int, Int),
        padding: Padding
    ) {
        self.poolSize = poolSize
        self.strides = strides
        self.padding = padding
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func call(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.averagePooled2D(kernelSize: poolSize, strides: strides, padding: padding)
    }
}

public extension AvgPool2D {
    /// Creates an average pooling layer.
    ///
    /// - Parameters:
    ///   - poolSize: Vertical and horizontal factors by which to downscale.
    ///   - strides: The strides.
    ///   - padding: The padding.
    init(poolSize: (Int, Int), strides: (Int, Int), padding: Padding = .valid) {
        self.init(poolSize: (1, poolSize.0, poolSize.1, 1),
                  strides: (1, strides.0, strides.1, 1),
                  padding: padding)
  }
}

/// An average pooling layer for spatial or spatio-temporal data.
@_fixed_layout
public struct AvgPool3D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The size of the sliding reduction window for pooling.
    @noDerivative let poolSize: (Int, Int, Int, Int, Int)
    /// The strides of the sliding window for each dimension of a 5-D input.
    /// Strides in non-spatial dimensions must be `1`.
    @noDerivative let strides: (Int, Int, Int, Int, Int)
    /// The padding algorithm for pooling.
    @noDerivative let padding: Padding

    /// Creates an average pooling layer.
    public init(
        poolSize: (Int, Int, Int, Int, Int),
        strides: (Int, Int, Int, Int, Int),
        padding: Padding
    ) {
        self.poolSize = poolSize
        self.strides = strides
        self.padding = padding
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func call(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.averagePooled3D(kernelSize: poolSize, strides: strides, padding: padding)
    }
}

public extension AvgPool3D {
    /// Creates an average pooling layer.
    ///
    /// - Parameters:
    ///   - poolSize: Vertical and horizontal factors by which to downscale.
    ///   - strides: The strides.
    ///   - padding: The padding.
    init(poolSize: (Int, Int, Int), strides: (Int, Int, Int), padding: Padding = .valid) {
        self.init(poolSize: (1, poolSize.0, poolSize.1, poolSize.2, 1),
                  strides: (1, strides.0, strides.1, strides.2, 1),
                  padding: padding)
  }
}

public extension AvgPool3D {
    /// Creates an average pooling layer with the specified pooling window size and stride. All
    /// pooling sizes and strides are the same.
    init(poolSize: Int, strides: Int, padding: Padding = .valid) {
        self.init(poolSize: (poolSize, poolSize, poolSize),
                  strides: (strides, strides, strides),
                  padding: padding)
    }
}

/// A global average pooling layer for temporal data.
@_fixed_layout
public struct GlobalAvgPool1D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// Creates a global average pooling layer.
    public init() {}

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
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
    /// - Parameter input: The input to the layer.
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
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func call(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.mean(squeezingAxes: [1, 2, 3])
    }
}
