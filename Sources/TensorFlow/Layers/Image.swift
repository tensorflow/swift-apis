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
    /// The number of units to crop from the beginning of an input's temporal dimension.
    @noDerivative public let begin: Int
    /// The number of units to crop from the end of an input's temporal dimension.
    @noDerivative public let end: Int

    /// Creates a cropping layer to trim the temporal dimension.
    ///
    /// - Parameters:
    ///   - begin: The number of units to be trimmed off the beginning of the temporal dimension.
    ///   - end: The number of units to be trimmed off the end of the temporal dimension.
    public init(begin: Int = 1, end: Int = 1) {
        self.begin = begin
        self.end = end
    }

    /// Returns the cropped input tensor according to the `cropping` dimensions specified
    /// at initialization.
    ///
    /// - Parameter input: The 3D tensor to be cropped. Note the expected shape of the input is
    ///   `[batch size, axis to crop, features]`.
    /// - Returns: The cropped 3D tensor of shape `[batch, cropped axis, feature]`.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.slice(
            lowerBounds: [0, begin, 0],
            upperBounds: [input.shape[0], input.shape[1] - end, input.shape[2]])
    }
}

/// A layer for cropping tensors along spatial dimensions, e.g. image cropping.
///
/// `Cropping2D` can trim an input at the top, bottom, left, and right sides of
/// its spatial dimensions.
public struct Cropping2D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The number of units to trim off the top of an input's height dimension.
    @noDerivative public let top: Int
    /// The number of units to trim off the bottom of an input's height dimension.
    @noDerivative public let bottom: Int
    /// The number of units to trim off from the left side of an input's width dimension.
    @noDerivative public let left: Int
    /// The number of units to trim off from the right side of an input's width dimension.
    @noDerivative public let right: Int

    /// Creates a cropping layer to trim spatial dimensions, i.e. height and width.
    ///
    /// - Parameters:
    ///   - top: The number of units to trim off the top of an input's height dimension.
    ///   - bottom: The number of units to trim off the bottom of an input's height dimension.
    ///   - left: The number of units to trim off the left of an input's height dimension.
    ///   - right: The number of units to trim off the right of an input's height dimension.
    // TODO: Add data format property support when control flow differentiation is completed.
    public init(top: Int = 1, bottom: Int = 1, left: Int = 1, right: Int = 1 ) {
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
    }

    /// Creates a cropping layer to trim spatial dimensions, i.e. height and width.
    ///
    /// - Parameters
    ///   - symmetricHeight: The number of units to trim off the top and bottom of an input's height
    ///   dimension.
    ///   - symmetricWidth: The number of units to trim off the left and right of an input's width
    ///   dimension.
    public init(symmetricHeight: Int = 1, symmetricWidth: Int = 1) {
        self.init(top: symmetricHeight, bottom: symmetricHeight,
            left: symmetricWidth, right: symmetricWidth)
    }

    /// Returns the cropped input tensor according to the `cropping` dimensions specified
    /// at initialization.
    ///
    /// - Parameter input: The 4D tensor to be cropped. Note that the expected data format is 
    ///   channel-last, i.e. shape `[batch size, height, width, channel count]`.
    /// - Returns: The cropped tensor.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.slice(
            lowerBounds: [0, top, left, 0],
            upperBounds: [input.shape[0], input.shape[1] - bottom,
                input.shape[2] - right, input.shape[3]])
    }
}
