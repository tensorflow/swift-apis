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

/// An upsampling layer for 1-D inputs.
@_fixed_layout
public struct UpSampling1D<Scalar: TensorFlowFloatingPoint>: Layer {
    @noDerivative public let size: Int

    /// Creates an upsampling layer.
    ///
    /// - Parameter size: The upsampling factor for timesteps.
    public init(size: Int) {
       self.size = size
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func call(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let shape = input.shape
        let (batchSize, timesteps, channels) = (shape[0], shape[1], shape[2])
        let scaleOnes = Tensor<Scalar>(ones: [1, 1, size, 1])
        let upSampling = input.reshaped(to: [batchSize, timesteps, 1, channels]) * scaleOnes
        return upSampling.reshaped(to: [batchSize, timesteps * size, channels])
    }
}

/// An upsampling layer for 2-D inputs.
@_fixed_layout
public struct UpSampling2D<Scalar: TensorFlowFloatingPoint>: Layer {
    @noDerivative public let size: Int

    /// Creates an upsampling layer.
    ///
    /// - Parameter size: The upsampling factor for rows and columns.
    public init(size: Int) {
       self.size = size
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func call(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let shape = input.shape
        let (batchSize, height, width, channels) = (shape[0], shape[1], shape[2], shape[3])
        let scaleOnes = Tensor<Scalar>(ones: [1, 1, size, 1, size, 1])
        let upSampling = input.reshaped(to: [batchSize, height, 1, width, 1, channels]) * scaleOnes
        return upSampling.reshaped(to: [batchSize, height * size, width * size, channels])
    }
}

/// An upsampling layer for 3-D inputs.
@_fixed_layout
public struct UpSampling3D<Scalar: TensorFlowFloatingPoint>: Layer {
    @noDerivative public let size: Int

    /// Creates an upsampling layer.
    ///
    /// - Parameter size: The upsampling factor for rows and columns.
    public init(size: Int) {
       self.size = size
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func call(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let shape = input.shape
        let (batchSize, height, width, depth, channels) =
            (shape[0], shape[1], shape[2], shape[3], shape[4])
        let scaleOnes = Tensor<Scalar>(ones: [1, 1, size, 1, size, 1, size, 1])
        let upSampling = input.reshaped(
            to: [batchSize, height, 1, width, 1, depth, 1, channels]) * scaleOnes
        return upSampling.reshaped(
            to: [batchSize, height * size, width * size, depth * size, channels])
    }
}
