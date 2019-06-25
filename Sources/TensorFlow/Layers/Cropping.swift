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

/// A layer for cropping tensors along the temporal dimension.
public struct Cropping1D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The cropping dimensions along the temporal axis.
    @noDerivative public let cropping: (Int, Int)

    /// Creates a cropping layer to trim the temporal dimension.
    ///
    /// - Parameter cropping: A tuple of two integers describing how many units should be trimmed
    ///   off the beginning and end of the cropping dimension.
    public init(cropping: (Int, Int) = (1, 1)) {
        self.cropping = cropping
    }

    /// Returns the cropped input tensor according to the `cropping` dimensions specified
    /// at initialization.
    ///
    /// - Parameter input: The 3D tensor to be cropped. Note the expected shape of the input is
    ///   `[batch size, axis to crop, features]`.
    /// - Returns: The cropped 3D tensor of shape `[batch, cropped axis, feature]`.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let (start, end) = cropping
        return input.slice(
            lowerBounds: [0, start, 0],
            upperBounds: [input.shape[0], input.shape[1] - end, input.shape[2]])
    }
}

/// A layer for cropping tensors along spatial dimensions, e.g. image cropping.
///
/// `Cropping2D` can trim an input at the top, bottom, left, and right sides of
/// its spatial dimensions.
public struct Cropping2D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The cropping dimensions along the height and width axes.
    @noDerivative public let cropping: ((Int, Int), (Int, Int))

    /// Creates a cropping layer to trim spatial dimensions, i.e. height and width.
    ///
    /// - Parameter cropping: A tuple of two tuples containing two integers describing how many
    ///   units should be trimmed off the height and width dimensions,
    ///   i.e. `((top, bottom), (left, right))`.
    // TODO: Add data format property support when control flow differentiation is completed.
    public init(cropping: ((Int, Int), (Int, Int)) = ((0, 0), (0, 0))) {
        self.cropping = cropping
    }

    /// Creates a cropping layer to trim spatial dimensions, i.e. height and width.
    ///
    /// - Parameter cropping: A tuple of two integers describing how many units should be
    ///   symmetrically trimmed off the height and width dimensions,
    ///   i.e. `((height, height), (width, width))`.
    public init(cropping: (Int, Int) = (0, 0)) {
        let (height, width) = cropping
        self.init(cropping: ((height, height), (width, width)))
    }

    /// Returns the cropped input tensor according to the `cropping` dimensions specified
    /// at initialization.
    ///
    /// - Parameter input: The 4D tensor to be cropped. Note that the expected data format is 
    ///   channel-last, i.e. shape `[batch size, height, width, channel count]`.
    /// - Returns: The cropped tensor.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let (top, bottom) = cropping.0
        let (left, right) = cropping.1
        return input.slice(
            lowerBounds: [0, top, left, 0],
            upperBounds: [input.shape[0], input.shape[1] - bottom,
                input.shape[2] - right, input.shape[3]])
    }
}
