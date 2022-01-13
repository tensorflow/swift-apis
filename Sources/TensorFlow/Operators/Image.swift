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

import _Differentiation

/// A resize algorithm.
public enum ResizeMethod {
  /// Nearest neighbor interpolation.
  case nearest
  /// Bilinear interpolation.
  case bilinear
  /// Bicubic interpolation.
  case bicubic
  /// Lanczos kernel with radius `3`.
  case lanczos3
  /// Lanczos kernel with radius `5`.
  case lanczos5
  /// Gaussian kernel with radius `3`, sigma `1.5 / 3.0`.
  case gaussian
  /// Mitchell-Netravali Cubic non-interpolating filter.
  case mitchellcubic
}

/// Resize images to size using the specified method.
///
/// - Parameters:
///   - images: 4-D `Tensor` of shape `[batch, height, width, channels]` or 3-D `Tensor` of shape `[height, width, channels]`.
///   - size: The new size of the images.
///   - method: The resize method. The default value is `.bilinear`.
///   - antialias: Iff `true`, use an anti-aliasing filter when downsampling an image.
/// - Precondition: The images must have rank `3` or `4`.
/// - Precondition: The size must be positive.
@differentiable(reverse, wrt: images)
public func resize(
  images: Tensor<Float>,
  size: (newHeight: Int, newWidth: Int),
  method: ResizeMethod = .bilinear,
  antialias: Bool = false
) -> Tensor<Float> {
  precondition(
    images.rank == 3 || images.rank == 4,
    "The images tensor must have rank 3 or 4.")
  precondition(size.newHeight > 0 && size.newWidth > 0, "The size must be positive.")
  var images = images
  let singleImage = images.rank == 3
  if singleImage {
    images = images.rankLifted()
  }
  let size = Tensor([Int32(size.newHeight), Int32(size.newWidth)], on: .defaultTFEager)
  let scale =
    Tensor<Float>(size)
    / Tensor<Float>([Float(images.shape[1]), Float(images.shape[2])], on: .defaultTFEager)
  switch method {
  case .nearest:
    images = resizeNearestNeighbor(
      images: images,
      size: size,
      halfPixelCenters: true)
  case .bilinear:
    if antialias {
      images = scaleAndTranslate(
        images: images,
        size: size,
        scale: scale,
        translation: Tensor(zeros: [2], on: .defaultTFEager),
        kernelType: "triangle")
    } else {
      images = resizeBilinear(
        images: images,
        size: size,
        halfPixelCenters: true)
    }
  case .bicubic:
    if antialias {
      images = scaleAndTranslate(
        images: images,
        size: size,
        scale: scale,
        translation: Tensor(zeros: [2], on: .defaultTFEager),
        kernelType: "keyscubic")
    } else {
      images = resizeBicubic(
        images: images,
        size: size,
        halfPixelCenters: true)
    }
  case .lanczos3:
    images = scaleAndTranslate(
      images: images,
      size: size,
      scale: scale,
      translation: Tensor(zeros: [2], on: .defaultTFEager),
      kernelType: "lanczos3",
      antialias: antialias)
  case .lanczos5:
    images = scaleAndTranslate(
      images: images,
      size: size,
      scale: scale,
      translation: Tensor(zeros: [2], on: .defaultTFEager),
      kernelType: "lanczos5",
      antialias: antialias)
  case .gaussian:
    images = scaleAndTranslate(
      images: images,
      size: size,
      scale: scale,
      translation: Tensor(zeros: [2], on: .defaultTFEager),
      kernelType: "gaussian",
      antialias: antialias)
  case .mitchellcubic:
    images = scaleAndTranslate(
      images: images,
      size: size,
      scale: scale,
      translation: Tensor(zeros: [2], on: .defaultTFEager),
      kernelType: "mitchellcubic",
      antialias: antialias)
  }
  if singleImage {
    images = images.squeezingShape(at: 0)
  }
  return images
}

/// Resize images to size using area interpolation.
///
/// - Parameters:
///   - images: 4-D `Tensor` of shape `[batch, height, width, channels]` or 3-D `Tensor` of shape `[height, width, channels]`.
///   - size: The new size of the images.
/// - Precondition: The images must have rank `3` or `4`.
/// - Precondition: The size must be positive.
@inlinable
public func resizeArea<Scalar: TensorFlowNumeric>(
  images: Tensor<Scalar>,
  size: (newHeight: Int, newWidth: Int),
  alignCorners: Bool = false
) -> Tensor<Float> {
  precondition(
    images.rank == 3 || images.rank == 4,
    "The images tensor must have rank 3 or 4.")
  precondition(size.newHeight > 0 && size.newWidth > 0, "The size must be positive.")
  var images = images
  let singleImage = images.rank == 3
  if singleImage {
    images = images.rankLifted()
  }
  let size = Tensor([Int32(size.newHeight), Int32(size.newWidth)], on: .defaultTFEager)
  var resized = _Raw.resizeArea(
    images: images,
    size: size,
    alignCorners: alignCorners)
  if singleImage {
    resized = resized.squeezingShape(at: 0)
  }
  return resized
}

@usableFromInline
@differentiable(reverse, wrt: images)
func scaleAndTranslate(
  images: Tensor<Float>,
  size: Tensor<Int32>,
  scale: Tensor<Float>,
  translation: Tensor<Float>,
  kernelType: String = "lanczos3",
  antialias: Bool = true
) -> Tensor<Float> {
  _Raw.scaleAndTranslate(
    images: images,
    size: size,
    scale: scale,
    translation: translation,
    kernelType: kernelType,
    antialias: antialias)
}

@usableFromInline
@derivative(of: scaleAndTranslate, wrt: images)
func _vjpScaleAndTranslate(
  images: Tensor<Float>,
  size: Tensor<Int32>,
  scale: Tensor<Float>,
  translation: Tensor<Float>,
  kernelType: String = "lanczos3",
  antialias: Bool = true
) -> (value: Tensor<Float>, pullback: (Tensor<Float>) -> Tensor<Float>) {
  let scaled = scaleAndTranslate(
    images: images,
    size: size,
    scale: scale,
    translation: translation,
    kernelType: kernelType,
    antialias: antialias)
  return (
    scaled,
    { v in
      _Raw.scaleAndTranslateGrad(
        grads: v,
        originalImage: images,
        scale: scale,
        translation: translation,
        kernelType: kernelType,
        antialias: antialias)
    }
  )
}

@usableFromInline
@differentiable(reverse, wrt: images where Scalar: TensorFlowFloatingPoint)
func resizeNearestNeighbor<Scalar: TensorFlowNumeric>(
  images: Tensor<Scalar>,
  size: Tensor<Int32>,
  alignCorners: Bool = false,
  halfPixelCenters: Bool = false
) -> Tensor<Scalar> {
  _Raw.resizeNearestNeighbor(
    images: images,
    size: size,
    alignCorners: alignCorners,
    halfPixelCenters: halfPixelCenters
  )
}

@usableFromInline
@derivative(of: resizeNearestNeighbor)
func _vjpResizeNearestNeighbor<Scalar: TensorFlowFloatingPoint>(
  images: Tensor<Scalar>,
  size: Tensor<Int32>,
  alignCorners: Bool,
  halfPixelCenters: Bool
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> Tensor<Scalar>) {
  let resized = resizeNearestNeighbor(
    images: images,
    size: size,
    alignCorners: alignCorners,
    halfPixelCenters: halfPixelCenters
  )
  return (
    resized,
    { v in
      _Raw.resizeNearestNeighborGrad(
        grads: v,
        size: Tensor([Int32(images.shape[1]), Int32(images.shape[2])], on: .defaultTFEager),
        alignCorners: alignCorners,
        halfPixelCenters: halfPixelCenters
      )
    }
  )
}

@usableFromInline
@differentiable(reverse, wrt: images where Scalar: TensorFlowFloatingPoint)
func resizeBilinear<Scalar: TensorFlowNumeric>(
  images: Tensor<Scalar>,
  size: Tensor<Int32>,
  alignCorners: Bool = false,
  halfPixelCenters: Bool = false
) -> Tensor<Float> {
  _Raw.resizeBilinear(
    images: images,
    size: size,
    alignCorners: alignCorners,
    halfPixelCenters: halfPixelCenters
  )
}

@usableFromInline
@derivative(of: resizeBilinear)
func _vjpResizeBilinear<Scalar: TensorFlowFloatingPoint>(
  images: Tensor<Scalar>,
  size: Tensor<Int32>,
  alignCorners: Bool,
  halfPixelCenters: Bool
) -> (value: Tensor<Float>, pullback: (Tensor<Float>) -> Tensor<Scalar>) {
  let resized = resizeBilinear(
    images: images,
    size: size,
    alignCorners: alignCorners,
    halfPixelCenters: halfPixelCenters
  )
  return (
    resized,
    { v in
      _Raw.resizeBilinearGrad(
        grads: v,
        originalImage: images,
        alignCorners: alignCorners,
        halfPixelCenters: halfPixelCenters
      )
    }
  )
}

@usableFromInline
@differentiable(reverse, wrt: images where Scalar: TensorFlowFloatingPoint)
func resizeBicubic<Scalar: TensorFlowFloatingPoint>(
  images: Tensor<Scalar>,
  size: Tensor<Int32>,
  alignCorners: Bool = false,
  halfPixelCenters: Bool = false
) -> Tensor<Float> {
  _Raw.resizeBicubic(
    images: images,
    size: size,
    alignCorners: alignCorners,
    halfPixelCenters: halfPixelCenters
  )
}

@usableFromInline
@derivative(of: resizeBicubic)
func _vjpResizeBicubic<Scalar: TensorFlowFloatingPoint>(
  images: Tensor<Scalar>,
  size: Tensor<Int32>,
  alignCorners: Bool,
  halfPixelCenters: Bool
) -> (value: Tensor<Float>, pullback: (Tensor<Float>) -> Tensor<Scalar>) {
  let resized = resizeBicubic(
    images: images,
    size: size,
    alignCorners: alignCorners,
    halfPixelCenters: halfPixelCenters
  )
  return (
    resized,
    { v in
      _Raw.resizeBicubicGrad(
        grads: v,
        originalImage: images,
        alignCorners: alignCorners,
        halfPixelCenters: halfPixelCenters
      )
    }
  )
}

/// Returns a 2-D dilation with the specified input, filter, strides, and padding.
///
/// - Parameters:
///   - input: The input.
///   - filter: The dilation filter.
///   - strides: The strides of the sliding filter for each dimension of the input.
///   - padding: The padding for the operation
///   - rates: The dilation rates for each dimension of the input.
/// - Precondition: `input` must have rank `4`.
/// - Precondition: `filter` must have rank `3`.
@differentiable(reverse, wrt: (input, filter))
public func dilation2D<Scalar: TensorFlowFloatingPoint>(
  _ input: Tensor<Scalar>,
  filter: Tensor<Scalar>,
  strides: (Int, Int, Int, Int) = (1, 1, 1, 1),
  rates: (Int, Int, Int, Int) = (1, 1, 1, 1),
  padding: Padding = .valid
) -> Tensor<Scalar> {
  precondition(input.shape.rank == 4, "The input must have rank 4.")
  precondition(filter.shape.rank == 3, "The filter must have rank 3.")
  return _Raw.dilation2D(
    input,
    filter: filter,
    strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3)],
    rates: [Int32(rates.0), Int32(rates.1), Int32(rates.2), Int32(rates.3)],
    padding: padding.raw
  )
}

@usableFromInline
@derivative(of: dilation2D)
func _vjpDilation2D<Scalar: TensorFlowFloatingPoint>(
  _ input: Tensor<Scalar>,
  filter: Tensor<Scalar>,
  strides: (Int, Int, Int, Int),
  rates: (Int, Int, Int, Int),
  padding: Padding
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
  let value = dilation2D(input, filter: filter, strides: strides, rates: rates, padding: padding)
  return (
    value,
    { v in
      (
        dilation2DBackpropInput(
          v, input: input, filter: filter,
          strides: strides, rates: rates, padding: padding),
        dilation2DBackpropFilter(
          v, input: input, filter: filter,
          strides: strides, rates: rates, padding: padding)
      )
    }
  )
}

/// TensorFlow builtin dilation2d gradient helper for the input.
@differentiable(reverse, wrt: (x, filter))
@usableFromInline
func dilation2DBackpropInput<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  input: Tensor<Scalar>,
  filter: Tensor<Scalar>,
  strides: (Int, Int, Int, Int) = (1, 1, 1, 1),
  rates: (Int, Int, Int, Int) = (1, 1, 1, 1),
  padding: Padding = .valid
) -> Tensor<Scalar> {
    return _Raw.dilation2DBackpropInput(
    input,
    filter: filter,
    outBackprop: x,
    strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3)],
    rates: [Int32(rates.0), Int32(rates.1), Int32(rates.2), Int32(rates.3)],
    padding: padding.raw)
}

@usableFromInline
@derivative(of: dilation2DBackpropInput, wrt: (x, filter))
func _vjpDilation2DBackpropInput<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  input: Tensor<Scalar>,
  filter: Tensor<Scalar>,
  strides: (Int, Int, Int, Int) = (1, 1, 1, 1),
  rates: (Int, Int, Int, Int) = (1, 1, 1, 1),
  padding: Padding = .valid
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
    let value = dilation2DBackpropInput(
            x,
            input: input,
            filter: filter,
            strides: strides,
            rates: rates,
            padding: padding)
    return (
      value,
      { v in
        (
          dilation2D(v, filter: filter, strides: strides, rates: rates, padding: padding),
          dilation2DBackpropFilter(
            x, input: v, filter: filter, strides: strides, rates: rates,
            padding: padding)
        )
      }
    )
}

/// TensorFlow builtin dilation2d gradient helper for the input.
@differentiable(reverse, wrt: (x, input))
@usableFromInline
func dilation2DBackpropFilter<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  input: Tensor<Scalar>,
  filter: Tensor<Scalar>,
  strides: (Int, Int, Int, Int) = (1, 1, 1, 1),
  rates: (Int, Int, Int, Int) = (1, 1, 1, 1),
  padding: Padding = .valid
) -> Tensor<Scalar> {
  return _Raw.dilation2DBackpropFilter(
    input,
    filter: filter,
    outBackprop: x,
    strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3)],
    rates: [Int32(rates.0), Int32(rates.1), Int32(rates.2), Int32(rates.3)],
    padding: padding.raw)
}

@usableFromInline
@derivative(of: dilation2DBackpropFilter, wrt:(x, input))
func _vjpDilation2DBackpropFilter<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  input: Tensor<Scalar>,
  filter: Tensor<Scalar>,
  strides: (Int, Int, Int, Int) = (1, 1, 1, 1),
  rates: (Int, Int, Int, Int) = (1, 1, 1, 1),
  padding: Padding = .valid
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
    let value = dilation2DBackpropFilter(
            x,
            input: input,
            filter: filter,
            strides: strides,
            rates: rates,
            padding: padding)
    return (
      value,
      { v in
        (
          dilation2D(input, filter: v, strides: strides, rates: rates, padding: padding),
          dilation2DBackpropInput(
            x, input: input, filter: v, strides: strides, rates: rates,
            padding: padding)
        )
      }
    )
}

/// Returns a 2-D erosion with the specified input, filter, strides, and padding.
///
/// - Parameters:
///   - input: The input.
///   - filter: The erosion filter.
///   - strides: The strides of the sliding filter for each dimension of the input.
///   - padding: The padding for the operation
///   - rates: The dilation rates for each dimension of the input.
/// - Precondition: `input` must have rank `4`.
/// - Precondition: `filter` must have rank 3.
@differentiable(reverse, wrt: (input, filter))
public func erosion2D<Scalar: TensorFlowFloatingPoint>(
  _ input: Tensor<Scalar>,
  filter: Tensor<Scalar>,
  strides: (Int, Int, Int, Int) = (1, 1, 1, 1),
  rates: (Int, Int, Int, Int) = (1, 1, 1, 1),
  padding: Padding = .valid
) -> Tensor<Scalar> {
  precondition(input.shape.rank == 4, "The input must have rank 4.")
  precondition(filter.shape.rank == 3, "The filter must have rank 3.")
  return -_Raw.dilation2D(
    -input,
    filter: filter.reversed(inAxes: [0, 1]),
    strides: [Int32(strides.0), Int32(strides.1), Int32(strides.2), Int32(strides.3)],
    rates: [Int32(rates.0), Int32(rates.1), Int32(rates.2), Int32(rates.3)],
    padding: padding.raw
  )
}

@usableFromInline
@derivative(of: erosion2D)
func _vjpErosion2D<Scalar: TensorFlowFloatingPoint>(
  _ input: Tensor<Scalar>,
  filter: Tensor<Scalar>,
  strides: (Int, Int, Int, Int),
  rates: (Int, Int, Int, Int),
  padding: Padding
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
  let negatedInput = -input
  let reversedFilter = filter.reversed(inAxes: [0, 1])
  let value = erosion2D(input, filter: filter, strides: strides, rates: rates, padding: padding)
  return (
    value,
    { v in
      (
        dilation2DBackpropInput(
          v, input: negatedInput, filter: reversedFilter,
          strides: strides, rates: rates, padding: padding),
        -dilation2DBackpropFilter(
          v, input: negatedInput, filter: reversedFilter,
            strides: strides, rates: rates, padding: padding).reversed(inAxes: [0, 1])
      )
    }
  )
}
