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

/// An upsampling layer for 1-D inputs.
@frozen
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
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let shape = input.shape
        let (batchSize, timesteps, channels) = (shape[0], shape[1], shape[2])
        let scaleOnes = Tensor<Scalar>(ones: [1, 1, size, 1])
        let upSampling = input.reshaped(to: [batchSize, timesteps, 1, channels]) * scaleOnes
        return upSampling.reshaped(to: [batchSize, timesteps * size, channels])
    }
}

/// An upsampling layer for 2-D inputs.
@frozen
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
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let shape = input.shape
        let (batchSize, height, width, channels) = (shape[0], shape[1], shape[2], shape[3])
        let scaleOnes = Tensor<Scalar>(ones: [1, 1, size, 1, size, 1])
        let upSampling = input.reshaped(to: [batchSize, height, 1, width, 1, channels]) * scaleOnes
        return upSampling.reshaped(to: [batchSize, height * size, width * size, channels])
    }
}

/// An upsampling layer for 3-D inputs.
@frozen
public struct UpSampling3D<Scalar: TensorFlowFloatingPoint>: Layer {
    @noDerivative public let size: Int

    /// Creates an upsampling layer.
    ///
    /// - Parameter size: The upsampling factor for rows and columns.
    public init(size: Int) {
        self.size = size
    }

    /// Repeats the elements of a tensor along an axis, like `np.repeat`.
    /// Function adapted from `def repeat_elements`:
    /// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/backend.py
    @differentiable(vjp: _vjpRepeatingElements)
    private func repeatingElements(
        _ input: Tensor<Scalar>, alongAxis axis: Int, count: Int
    ) -> Tensor<Scalar> {
        let splits = Raw.split(splitDim: Tensor<Int32>(Int32(axis)),
                               value: input, numSplit: Int64(input.shape[axis]))
        let repeated = splits.flatMap { x in Array(repeating: x, count: count) }
        return Tensor<Scalar>(concatenating: repeated, alongAxis: axis)
    }

    private func _vjpRepeatingElements(
        _ input: Tensor<Scalar>, alongAxis axis: Int, count: Int
    ) -> (Tensor<Scalar>, (Tensor<Scalar>) -> (AllDifferentiableVariables, Tensor<Scalar>)) {
        let value = repeatingElements(input, alongAxis: axis, count: count)
        return (value, { v in
            let splits = Raw.split(splitDim: Tensor<Int32>(Int32(axis)),
                                   value: v, numSplit: Int64(input.shape[axis]))
            let summed = splits.map { x in x.sum(alongAxes: axis) }
            let concatenated = Tensor<Scalar>(concatenating: summed, alongAxis: axis)
            return (.zero, concatenated)
        })
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        var result = repeatingElements(input, alongAxis: 1, count: size)
        result = repeatingElements(result, alongAxis: 2, count: size)
        result = repeatingElements(result, alongAxis: 3, count: size)
        return result
    }
}
