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

// MARK: - Matrix operations

public extension Tensor where Scalar: TensorFlowNumeric {
    /// Returns the [batched] diagonal part of a [batched] tensor.
    /// For the tensor instance of the shape `[..., M, N]`, the output is a tensor
    /// of the shape `[..., K]`, where `K` equals `min(N, M)`.
    ///
    /// For example:
    ///
    /// ```
    /// // 't' is [[1, 0, 0, 0]
    /// //         [0, 2, 0, 0]
    /// //         [0, 0, 3, 0]
    /// //         [0, 0, 0, 4]]
    /// t.diagonalPart()
    /// // [1, 2, 3, 4]
    /// ```
    @inlinable
    @differentiable(wrt: self, vjp: _vjpDiagonalPart where Scalar: TensorFlowFloatingPoint)
    func diagonalPart() -> Tensor {
        precondition(rank >= 2, "The tensor must have at least rank 2.")
        return _Raw.matrixDiagPart(self)
    }

    /// Constructs a [batched] diagonal array.
    /// For the tensor instance of the shape `[..., M]`, the output is a tensor of the shape `[..., M, M]`.
    ///
    /// For example:
    ///
    /// ```
    /// // 't' is [1, 2, 3, 4]
    ///
    /// t.diagonal()
    /// // [[1, 0, 0, 0]
    /// //  [0, 2, 0, 0]
    /// //  [0, 0, 3, 0]
    /// //  [0, 0, 0, 4]]
    /// ```
    @inlinable
    @differentiable(wrt: self, vjp: _vjpDiagonal where Scalar: TensorFlowFloatingPoint)
    func diagonal() -> Tensor {
        _Raw.matrixDiag(diagonal: self)
    }

    /// Returns a copy of a innermost tensor defined by a central band boundaries.
    /// The output is a tensor of the same shape as the instance `[..., :, :]`.
    ///
    /// For example:
    ///
    /// ```
    /// // 't' is [[ 0,  1,  2, 3]
    /// //         [-1,  0,  1, 2]
    /// //         [-2, -1,  0, 1]
    /// //         [-3, -2, -1, 0]]
    ///
    /// t.bandPart(1, -1)
    /// // [[ 0,  1,  2, 3]
    /// //  [-1,  0,  1, 2]
    /// //  [ 0, -1,  0, 1]
    /// //  [ 0,  0, -1, 0]]
    ///
    /// t.bandPart(2, 1)
    /// // [[ 0,  1,  0, 0]
    /// //  [-1,  0,  1, 0]
    /// //  [-2, -1,  0, 1]
    /// //  [ 0, -2, -1, 0]]
    /// ```
    @inlinable
    @differentiable(wrt: self, vjp: _vjpBandPart where Scalar: TensorFlowFloatingPoint)
    func bandPart(_ lowerCount: Int, _ upperCount: Int) -> Tensor {
        precondition(rank >= 2, "The tensor must have at least rank 2.")
        let lower = Tensor<Int32>(Int32(lowerCount))
        let upper = Tensor<Int32>(Int32(upperCount))
        return _Raw.matrixBandPart(self, numLower: lower, numUpper: upper)
    }
}

internal extension Tensor where Scalar: TensorFlowFloatingPoint {
    @inlinable
    func _vjpDiagonalPart() -> (Tensor, (Tensor) -> Tensor) {
        (diagonalPart(), { $0.diagonal() })
    }

    @inlinable
    func _vjpDiagonal() -> (Tensor, (Tensor) -> Tensor) {
        (diagonal(), { $0.diagonalPart() })
    }

    @inlinable
    func _vjpBandPart(_ numLower: Int, _ numUpper: Int) -> (Tensor, (Tensor) -> Tensor) {
        (bandPart(numLower, numUpper), { $0.bandPart(numLower, numUpper) })
    }
}

// MARK: - Decompositions

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

public extension Tensor where Scalar: TensorFlowFloatingPoint {
    /// Returns the QR decomposition of each inner matrix in the tensor, a tensor with inner
    /// orthogonal matrices `q` and a tensor with inner upper triangular matrices `r`, such that the
    /// tensor is equal to `matmul(q, r)`.
    ///
    /// - Parameters:
    ///   - fullMatrices: If `true`, compute full-sized `q` and `r`. Otherwise compute only the
    ///     leading `min(shape[rank - 1], shape[rank - 2])` columns of `q`.
    ///
    @inlinable
    func qrDecomposition(fullMatrices: Bool = false) -> (q: Tensor<Scalar>, r: Tensor<Scalar>) {
        _Raw.qr(self, fullMatrices: fullMatrices)
    }
}
