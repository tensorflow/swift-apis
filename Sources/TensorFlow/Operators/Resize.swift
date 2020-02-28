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
    case area, nearest, bilinear, bicubic, lanczos3, lanczos5, gaussian, mitchellcubic
}

@differentiable(wrt: images)
public func resize(
    images: Tensor<Float>,
    size: Tensor<Int32>,
    method: ResizeMethod,
    antialias: Bool = false
) -> Tensor<Float> {
    let scale = Tensor<Float>(size) / Tensor<Float>([Float(images.shape[1]), Float(images.shape[2])])
    
    switch method {
    case .area:
        return resizeArea(images: images, size: size)
    case .nearest:
        return resizeNearestNeighbor(images: images, size: size, halfPixelCenters: true)
    case .bilinear:
        if antialias {
            return scaleAndTranslate(images: images, size: size, scale: scale, translation: Tensor(zeros: [2]), kernelType: "triangle")
        } else {
            return resizeBilinear(images: images, size: size, halfPixelCenters: true)
        }
    case .bicubic:
        if antialias {
            return scaleAndTranslate(images: images, size: size, scale: scale, translation: Tensor(zeros: [2]), kernelType: "keyscubic")
        } else {
            return resizeBicubic(images: images, size: size, halfPixelCenters: true)
        }
    case .lanczos3:
        return scaleAndTranslate(images: images, size: size, scale: scale, translation: Tensor(zeros: [2]), kernelType: "lanczos3", antialias: antialias)
    case .lanczos5:
        return scaleAndTranslate(images: images, size: size, scale: scale, translation: Tensor(zeros: [2]), kernelType: "lanczos5", antialias: antialias)
    case .gaussian:
        return scaleAndTranslate(images: images, size: size, scale: scale, translation: Tensor(zeros: [2]), kernelType: "gaussian", antialias: antialias)
    case .mitchellcubic:
        return scaleAndTranslate(images: images, size: size, scale: scale, translation: Tensor(zeros: [2]), kernelType: "mitchellcubic", antialias: antialias)
    }
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
            scale: scaled,
            translation: translation,
            kernelType: kernelType,
            antialias: antialias)
    })
}

@differentiable(wrt: images)
func resizeArea(
    images: Tensor<Float>,
    size: Tensor<Int32>,
    alignCorners: Bool = false
) -> Tensor<Float> {
    _Raw.resizeArea(images: images,
                    size: size,
                    alignCorners: alignCorners)
}

@derivative(of: resizeArea)
func _vjpResizeArea(
    images: Tensor<Float>,
    size: Tensor<Int32>,
    alignCorners: Bool = false
) -> (value: Tensor<Float>, pullback: (Tensor<Float>)->Tensor<Float>) {
    let resized = resizeArea(images: images, size: size, alignCorners: alignCorners)
    return (resized, { v in
        let factor = Float(v.shape[1]*v.shape[2]) / Float(images.shape[1]*images.shape[2])
        return resizeArea(
            images: v,
            size: Tensor<Int32>([Int32(images.shape[1]), Int32(images.shape[2])]),
            alignCorners: alignCorners) * factor
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
            size: size,
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
