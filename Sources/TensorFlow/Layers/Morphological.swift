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

/// A 2-D morphological dilation layer
///
/// This layer returns the morphogical dilation of the input tensor with the provided filters
@frozen
public struct `Dilation2D`<Scalar: TensorFlowFloatingPoint>: Layer {
  /// The 4-D dilation filter.
  public var filter: Tensor<Scalar>
  /// The strides of the sliding window for spatial dimensions.
  @noDerivative public let strides: (Int, Int)
  /// The padding algorithm for dilation.
  @noDerivative public let padding: Padding
  /// The dilation factor for spatial dimensions.
  @noDerivative public let rates: (Int, Int)

  /// Creates a `Dilation2D` layer with the specified filter, strides,
  /// dilations and padding.
  ///
  /// - Parameters:
  ///   - filter: The 4-D dilation filter of shape
  ///     [filter height, filter width, input channel count, output channel count].
  ///   - strides: The strides of the sliding window for spatial dimensions, i.e.
  ///     (stride height, stride width).
  ///   - rates: The dilation rates for spatial dimensions, i.e.
  ///     (dilation height, dilation width).
  ///   - padding: The padding algorithm for dilation.
  public init(
    filter: Tensor<Scalar>,
    strides: (Int, Int) = (1, 1),
    rates: (Int, Int) = (1, 1),
    padding: Padding = .valid
  ) {
    self.filter = filter
    self.strides = strides
    self.padding = padding
    self.rates = rates
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// The output spatial dimensions are computed as:
  ///
  /// output height =
  /// (input height + 2 * padding height - (dilation height * (filter height - 1) + 1))
  /// / stride height + 1
  ///
  /// output width =
  /// (input width + 2 * padding width - (dilation width * (filter width - 1) + 1))
  /// / stride width + 1
  ///
  /// and padding sizes are determined by the padding scheme.
  ///
  /// - Parameter input: The input to the layer of shape
  ///   [batch size, input height, input width, input channel count].
  /// - Returns: The output of shape
  ///   [batch count, output height, output width, output channel count].
  ///
  /// - Note: Padding size equals zero when using `.valid`.
  @differentiable
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let dilated = dilation2D(
      input,
      filter: filter,
      strides: (1, strides.0, strides.1, 1),
      rates: (1, rates.0, rates.1, 1),
      padding: padding)

    return dilated
  }
}

/// A 2-D morphological erosion layer
///
/// This layer returns the morphogical erosion of the input tensor with the provided filters
@frozen
public struct `Erosion2D`<Scalar: TensorFlowFloatingPoint>: Layer {
  /// The 4-D dilation filter.
  public var filter: Tensor<Scalar>
  /// The strides of the sliding window for spatial dimensions.
  @noDerivative public let strides: (Int, Int)
  /// The padding algorithm for dilation.
  @noDerivative public let padding: Padding
  /// The dilation factor for spatial dimensions.
  @noDerivative public let rates: (Int, Int)

  /// Creates a `Erosion2D` layer with the specified filter, strides,
  /// dilations and padding.
  ///
  /// - Parameters:
  ///   - filter: The 4-D dilation filter of shape
  ///     [filter height, filter width, input channel count, output channel count].
  ///   - strides: The strides of the sliding window for spatial dimensions, i.e.
  ///     (stride height, stride width).
  ///   - rates: The dilation rates for spatial dimensions, i.e.
  ///     (dilation height, dilation width).
  ///   - padding: The padding algorithm for dilation.
  public init(
    filter: Tensor<Scalar>,
    strides: (Int, Int) = (1, 1),
    rates: (Int, Int) = (1, 1),
    padding: Padding = .valid
  ) {
    self.filter = filter
    self.strides = strides
    self.padding = padding
    self.rates = rates
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// The output spatial dimensions are computed as:
  ///
  /// output height =
  /// (input height + 2 * padding height - (dilation height * (filter height - 1) + 1))
  /// / stride height + 1
  ///
  /// output width =
  /// (input width + 2 * padding width - (dilation width * (filter width - 1) + 1))
  /// / stride width + 1
  ///
  /// and padding sizes are determined by the padding scheme.
  ///
  /// - Parameter input: The input to the layer of shape
  ///   [batch size, input height, input width, input channel count].
  /// - Returns: The output of shape
  ///   [batch count, output height, output width, output channel count].
  ///
  /// - Note: Padding size equals zero when using `.valid`.
  @differentiable
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let eroded = erosion2D(
      input,
      filter: filter,
      strides: (1, strides.0, strides.1, 1),
      rates: (1, rates.0, rates.1, 1),
      padding: padding)

    return eroded
  }
}
