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


/// Returns the Cholesky decomposition of one or more square matrices.
///
/// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
/// form square matrices.
///
/// The input has to be symmetric and positive definite. Only the lower-triangular
/// part of the input will be used for this operation. The upper-triangular part
/// will not be read.
///
/// The output is a tensor of the same shape as the input
/// containing the Cholesky decompositions for all input submatrices `[..., :, :]`.
///
/// - Parameter input: A tensor of shape `[..., M, M]`.
@inlinable
@differentiable(vjp: _vjpCholesky)
public func cholesky<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.cholesky(x)
}

@inlinable
internal func _vjpCholesky<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
    let decomposition = cholesky(x)
    return (decomposition, { v in _Raw.choleskyGrad(l: decomposition, grad: v)})
}

