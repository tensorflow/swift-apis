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

public enum ResizeMethod {
    case nearest, bilinear, bicubic, lanczos3, lanczos5, gaussian, mitchellcubic
}

/// Resize images to size using the specified method.
///
/// - Parameters:
///   - images: 4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of shape [height, width, channels].
///   - size: A tuple of two Int: (new_height, new_width). The new size for the images.
///   - method: ResizeMethod. Defaults to bilinear.
///   - antialias: Whether to use an anti-aliasing filter when downsampling an image.
/// - Precondition: The images must have rank 3 or 4.
@differentiable(wrt: images)
public func resize(
    images: Tensor<Float>,
    size: (Int, Int),
    method: ResizeMethod = .bilinear,
    antialias: Bool = false
) -> Tensor<Float> {
    precondition(images.rank == 3 || images.rank == 4,
                 "The images tensor must have rank 3 or 4.")
    var images = images
    let singleImage = images.rank == 3
    if singleImage {
        images = images.rankLifted()
    }
    let size = Tensor([Int32(size.0), Int32(size.1)])
    let scale = Tensor<Float>(size) / Tensor<Float>([Float(images.shape[1]), Float(images.shape[2])])
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
                translation: Tensor(zeros: [2]),
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
                translation: Tensor(zeros: [2]),
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
            translation: Tensor(zeros: [2]),
            kernelType: "lanczos3",
            antialias: antialias)
    case .lanczos5:
        images = scaleAndTranslate(
            images: images,
            size: size,
            scale: scale,
            translation: Tensor(zeros: [2]),
            kernelType: "lanczos5",
            antialias: antialias)
    case .gaussian:
        images = scaleAndTranslate(
            images: images,
            size: size,
            scale: scale,
            translation: Tensor(zeros: [2]),
            kernelType: "gaussian",
            antialias: antialias)
    case .mitchellcubic:
        images = scaleAndTranslate(
            images: images,
            size: size,
            scale: scale,
            translation: Tensor(zeros: [2]),
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
///   - images: 4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of shape [height, width, channels].
///   - size: A tuple of two Int: (new_height, new_width). The new size for the images.
/// - Precondition: The images must have rank 3 or 4.
@inlinable
public func resizeArea<Scalar: TensorFlowNumeric>(
    images: Tensor<Scalar>,
    size: (Int, Int),
    alignCorners: Bool = false
) -> Tensor<Float> {
    precondition(images.rank == 3 || images.rank == 4,
                 "The images tensor must have rank 3 or 4.")
    var images = images
    let singleImage = images.rank == 3
    if singleImage {
        images = images.rankLifted()
    }
    let size = Tensor([Int32(size.0), Int32(size.1)])
    
    var resized = _Raw.resizeArea(
        images: images,
        size: size,
        alignCorners: alignCorners)
    
    if singleImage {
        resized = resized.squeezingShape(at: 0)
    }
    
    return resized
}

@differentiable(wrt: images)
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
@derivative(of: scaleAndTranslate wrt: images)
func _vjpScaleAndTranslate(
    images: Tensor<Float>,
    size: Tensor<Int32>,
    scale: Tensor<Float>,
    translation: Tensor<Float>,
    kernelType: String = "lanczos3",
    antialias: Bool = true
) -> (value: Tensor<Float>, pullback: (Tensor<Float>)->Tensor<Float>) {
    let scaled = scaleAndTranslate(
        images: images,
        size: size,
        scale: scale,
        translation: translation,
        kernelType: kernelType,
        antialias: antialias)
    
    return (scaled, { v in
        _Raw.scaleAndTranslateGrad(
            grads: v,
            originalImage: images,
            scale: scale,
            translation: translation,
            kernelType: kernelType,
            antialias: antialias)
    })
}

@differentiable(wrt: images where Scalar: TensorFlowFloatingPoint)
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
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>)->Tensor<Scalar>) {
    let resized = resizeNearestNeighbor(
        images: images,
        size: size,
        alignCorners: alignCorners,
        halfPixelCenters: halfPixelCenters
    )
    return (resized, { v in
        _Raw.resizeNearestNeighborGrad(
            grads: v,
            size: Tensor([Int32(images.shape[1]), Int32(images.shape[2])]),
            alignCorners: alignCorners,
            halfPixelCenters: halfPixelCenters
        )
    })
}

@differentiable(wrt: images)
func resizeBilinear(
    images: Tensor<Float>,
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
func _vjpResizeBilinear(
    images: Tensor<Float>,
    size: Tensor<Int32>,
    alignCorners: Bool,
    halfPixelCenters: Bool
) -> (value: Tensor<Float>, pullback: (Tensor<Float>)->Tensor<Float>) {
    let resized = resizeBilinear(
        images: images,
        size: size,
        alignCorners: alignCorners,
        halfPixelCenters: halfPixelCenters
    )
    return (resized, { v in
        _Raw.resizeBilinearGrad(
            grads: v,
            originalImage: images,
            alignCorners: alignCorners,
            halfPixelCenters: halfPixelCenters
        )
    })
}

@differentiable(wrt: images)
func resizeBicubic(
    images: Tensor<Float>,
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
func _vjpResizeBicubic(
    images: Tensor<Float>,
    size: Tensor<Int32>,
    alignCorners: Bool,
    halfPixelCenters: Bool
) -> (value: Tensor<Float>, pullback: (Tensor<Float>)->Tensor<Float>) {
    let resized = resizeBicubic(
        images: images,
        size: size,
        alignCorners: alignCorners,
        halfPixelCenters: halfPixelCenters
    )
    return (resized, { v in
        _Raw.resizeBicubicGrad(
            grads: v,
            originalImage: images,
            alignCorners: alignCorners,
            halfPixelCenters: halfPixelCenters
        )
    })
}
