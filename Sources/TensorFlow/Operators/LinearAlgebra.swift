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
    @differentiable(where Scalar: TensorFlowFloatingPoint)
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
    @differentiable(where Scalar: TensorFlowFloatingPoint)
    func diagonal() -> Tensor {
        _Raw.matrixDiag(diagonal: self)
    }

    /// Returns `self` with new diagonal values, given that `self` is an optionally batched matrix.
    ///
    /// The returned tensor has the same shape and values as `self`, except for the specified
    /// diagonals of the innermost matrices which are overwritten by the values in `diagonal`.
    ///
    /// Parameter diagonal: A tensor with rank `rank - 1` representing the new diagonal values.
    @inlinable
    func withDiagonal(_ diagonal: Tensor<Scalar>) -> Tensor {
        _Raw.matrixSetDiag(self, diagonal: diagonal)
    }

    @available(*, deprecated, renamed: "bandPart(subdiagonalCount:superdiagonalCount:)")
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func bandPart(_ subdiagonalCount: Int, _ superdiagonalCount: Int) -> Tensor {
        return bandPart(subdiagonalCount: subdiagonalCount, superdiagonalCount: superdiagonalCount)
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
    ///
    /// - Parameters:
    ///   - subdiagonalCount: The number of subdiagonals to keep. If negative, keep entire lower
    ///     triangle.
    ///   - superdiagonalCount: The number of superdiagonals to keep. If negative, keep entire upper
    ///     triangle.
    @inlinable
    @differentiable(where Scalar: TensorFlowFloatingPoint)
    func bandPart(subdiagonalCount: Int, superdiagonalCount: Int) -> Tensor {
        precondition(rank >= 2, "The tensor must have at least rank 2.")
        let lower = Tensor<Int32>(Int32(subdiagonalCount))
        let upper = Tensor<Int32>(Int32(superdiagonalCount))
        return _Raw.matrixBandPart(self, numLower: lower, numUpper: upper)
    }
}

internal extension Tensor where Scalar: TensorFlowFloatingPoint {
    @inlinable
    @derivative(of: diagonalPart)
    func _vjpDiagonalPart() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        (diagonalPart(), { $0.diagonal() })
    }

    @inlinable
    @derivative(of: diagonal)
    func _vjpDiagonal() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        (diagonal(), { $0.diagonalPart() })
    }

    @inlinable
    @derivative(of: bandPart(subdiagonalCount:superdiagonalCount:))
    func _vjpBandPart(subdiagonalCount: Int, superdiagonalCount: Int) -> (
        value: Tensor, pullback: (Tensor) -> Tensor
    ) {
        let value = bandPart(
            subdiagonalCount: subdiagonalCount,
            superdiagonalCount: superdiagonalCount)
        return (value, {
            $0.bandPart(subdiagonalCount: subdiagonalCount, superdiagonalCount: superdiagonalCount)
        })
    }
}

/// Returns an identity matrix or a batch of matrices.
///
/// - Parameters:
///   - rowCount: The number of rows in each batch matrix.
///   - columnCount: The number of columns in each batch matrix.
///   - batchShape: The leading batch dimensions of the returned tensor.
public func eye<Scalar: Numeric>(
    rowCount: Int,
    columnCount: Int? = nil,
    batchShape: [Int] = []
) -> Tensor<Scalar> {
    let columnCount = columnCount ?? rowCount
    let diagonalSize = min(rowCount, columnCount)
    let diagonalShape = batchShape + [diagonalSize]
    let diagonalOnes = Tensor<Scalar>(ones: TensorShape(diagonalShape))
    if rowCount == columnCount {
        return diagonalOnes.diagonal()
    }
    let shape = batchShape + [rowCount, columnCount]
    let zeroMatrix = Tensor<Scalar>(zeros: TensorShape(shape))
    return zeroMatrix.withDiagonal(diagonalOnes)
}

/// Computes the trace of an optionally batched matrix.
/// The trace is the the sum along the main diagonal of each inner-most matrix.
///
/// The input is a tensor with shape `[..., M, N]`.
/// The output is a tensor with shape `[...]`.
///
/// - Parameter matrix: A tensor of shape `[..., M, N]`.
/// - Precondition: `matrix` must be a tensor with shape `[..., M, N]`.
@inlinable
@differentiable(wrt: matrix where T: TensorFlowFloatingPoint)
public func trace<T: TensorFlowNumeric>(_ matrix: Tensor<T>) -> Tensor<T> {
    precondition(matrix.rank >= 2, "The tensor must have at least rank 2.")
    return matrix.diagonalPart().sum(squeezingAxes: -1)
}

/// Computes the natural logarithm of the determinant of a hermitian positive definite matrix.
///
/// - Parameter matrix: A tensor of shape `[..., M, N]`.
/// - Returns: The natural logarithm of the determinant of `matrix`.
@inlinable
@differentiable(wrt: matrix where T: TensorFlowFloatingPoint)
func logdet<T: TensorFlowFloatingPoint>(_ matrix: Tensor<T>) -> Tensor<T> {
    return 2.0 * log(cholesky(matrix).diagonalPart()).sum(squeezingAxes: -1) 
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
@differentiable
public func cholesky<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
    _Raw.cholesky(x)
}

@inlinable
@derivative(of: cholesky)
internal func _vjpCholesky<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> Tensor<T>) {
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

    /// Returns the singular value decomposition of `self`, given that `self` is an optionally
    /// batched matrix.
    ///
    /// The singular value decomposition (SVD) of the optionally batched matrix `self` is values
    /// `s`, `u`, and `v`, such that:
    ///
    ///     self[..., :, :] = u[..., :, :] • s[..., :, :].diagonal() • v[..., :, :].transposed()`
    ///
    /// self` must be a tensor with shape `[..., M, N]`. Let `K = min(M, N)`.
    ///
    /// - Parameters:
    ///   - computeUV: If `true`, the left and right singular vectors are computed and returned as
    ///     `u` and `v`, respectively. If `false`, `nil` values are returned as `u` and `v`.
    ///   - fullMatrices: If `true`, `u` and `v` respectively have shapes `[..., M, M]` and
    ///     `[..., N, N]`. If `false`, `u` and `v` respectively have shapes `[..., M, K]` and
    ///     `[..., K, N]`. Ignored when `computeUV` is false.
    ///
    /// - Returns:
    ///   - s: The singular values, with shape `[..., K]`. Within each vector, the singular values
    ///     are sorted in descending order.
    ///   - u: The left singular vectors.
    ///   - v: The right singular vectors.
    ///
    /// - Precondition: `self` must be a tensor with shape `[..., M, N]`.
    @inlinable
    func svd(computeUV: Bool = true, fullMatrices: Bool = false) -> (
        s: Tensor<Scalar>, u: Tensor<Scalar>?, v: Tensor<Scalar>?
    ) {
        let (s, u, v) = _Raw.svd(self, computeUv: computeUV, fullMatrices: fullMatrices)
        if !computeUV {
            return (s, nil, nil)
        }
        return (s, u, v)
    }
}


// MARK: Solvers

/// Solves optionally batched systems of linear equations with upper or lower triangular matrices by backsubstitution.
/// Returns solution to the system `A x = b`. Shape of return matches `b`.
///
/// - Parameters:
///     - matrix: A batched matrix tensor.
///     - rhs: A batched vector tensor.
/// - Precondition: `matrix` must be a tensor with shape `[..., M, M]`.
/// - Precondition: `rhs` must be a tensor with shape `[..., M, K]`.
@inlinable
@differentiable
public func triangularSolve<T: TensorFlowFloatingPoint>(
    matrix: Tensor<T>,
    rhs: Tensor<T>,
    lower: Bool = true,
    adjoint: Bool = false
) -> Tensor<T> {
    precondition(matrix.rank >= 2, "The matrix tensor must have at least rank 2.")
    precondition(rhs.rank >= 2, "The rhs tensor must have at least rank 2.")
    if matrix.rank < rhs.rank {
        let leadingDims = extractLeadingDims(rhs.shape, matrix.shape, ignoreLast: 2)
        let matchedMatrix = matrix.broadcasted(to: leadingDims + matrix.shape)
        return _Raw.matrixTriangularSolve(matrix: matchedMatrix, rhs: rhs, lower: lower, adjoint: adjoint)
    } else if matrix.rank > rhs.rank {
        let leadingDims = extractLeadingDims(rhs.shape, matrix.shape, ignoreLast: 2)
        let matchedRhs = rhs.broadcasted(to: leadingDims + rhs.shape)
        return _Raw.matrixTriangularSolve(matrix: matrix, rhs: matchedRhs, lower: lower, adjoint: adjoint)
    }
    return _Raw.matrixTriangularSolve(matrix: matrix, rhs: rhs, lower: lower, adjoint: adjoint)
}

@inlinable
@derivative(of: triangularSolve)
internal func _vjpTriangularSolve<T: TensorFlowFloatingPoint>(
    matrix: Tensor<T>,
    rhs: Tensor<T>,
    lower: Bool = true,
    adjoint: Bool = false
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> (Tensor<T>, Tensor<T>)) {
    let broadcastMatrix: Bool = matrix.rank < rhs.rank
    let broadcastRhs: Bool = matrix.rank > rhs.rank

    var matchedMatrix: Tensor<T> = matrix
    if broadcastMatrix {
        let leadingDims = extractLeadingDims(rhs.shape, matrix.shape, ignoreLast: 2)
        matchedMatrix = matrix.broadcasted(to: leadingDims + matrix.shape)
    }
    
    let x = triangularSolve(matrix: matchedMatrix, rhs: rhs, lower: lower, adjoint: adjoint)
    let triangularSolveGrad = { (gradient: Tensor<T>) -> (Tensor<T>, Tensor<T>) in
        let rhsGrad = triangularSolve(matrix: matchedMatrix, rhs: gradient, lower: lower, adjoint: !adjoint)
        let (left, right) = adjoint ? (x, rhsGrad) : (rhsGrad, x)
        let matrixGrad = -matmul(left, transposed: false, right, transposed: true)
        let triMatrixGrad = lower
            ? matrixGrad.bandPart(subdiagonalCount: -1, superdiagonalCount: 0)
            : matrixGrad.bandPart(subdiagonalCount: 0, superdiagonalCount: -1)
        if broadcastMatrix {
            return (triMatrixGrad.unbroadcasted(to: matrix.shape), rhsGrad)
        } else if broadcastRhs {
            return (triMatrixGrad, rhsGrad.unbroadcasted(to: rhs.shape))
        }
        return (triMatrixGrad, rhsGrad)
    }
    return (x, triangularSolveGrad)
}


/// Returns leading dimension from tensors shapes.
/// Example:
/// The leading dimension for tensors with shapes `[N, M, K]` and `[M, K]`
/// is `[N]`. In cases, when tensors shapes do not have common parts from the end, e.g.
/// `[N, M, K]` and `[N, M]`, it will raise an error.
///
/// - Parameters:
///     - left: Left tensor.
///     - right: Right tensor.
///     - ignoreLast: Number of dimensions from the end to ignore. Default value is zero.
/// - Precondition: Left tensor rank cannot be lower than `ignoreLast`.
/// - Precondition: Right tensor rank cannot be lower than `ignoreLast`.
@inlinable
public func extractLeadingDims<T: TensorFlowNumeric>(
    _ left: Tensor<T>,
    _ right: Tensor<T>,
    ignoreLast: Int = 0
) -> TensorShape {
    extractLeadingDims(left.shape, right.shape, ignoreLast: ignoreLast)
}

/// Returns leading dimension from tensors shapes.
/// Example:
/// The leading dimension for tensors with shapes `[N, M, K]` and `[M, K]`
/// is `[N]`. In cases, when tensors shapes do not have common parts from the end, e.g.
/// `[N, M, K]` and `[N, M]`, it will raise an error.
///
/// - Parameters:
///     - left: Left tensor shape.
///     - right: Right tensor shape.
///     - ignoreLast: Number of dimensions from the end to ignore. Default value is zero.
/// - Precondition: Left tensor rank cannot be lower than `ignoreLast`.
/// - Precondition: Right tensor rank cannot be lower than `ignoreLast`.
@inlinable
public func extractLeadingDims(_ left: TensorShape, _ right: TensorShape, ignoreLast: Int = 0) -> TensorShape {
    precondition(left.rank >= ignoreLast, "The left tensor must have at least rank `ignoreLast`.")
    precondition(right.rank >= ignoreLast, "The right tensor must have at least rank 2.")
    let (smallerShape, largerShape) = left.rank > right.rank ? (right, left) : (left, right)
    let smaller = smallerShape.dropLast(ignoreLast)
    let larger = largerShape.dropLast(ignoreLast)
    let smallerEndIndex = smaller.count - 1
    let largerEndIndex = larger.count - 1
    for i in smaller.indices {
        internalConsistencyCheck(smaller[smallerEndIndex - i] == larger[largerEndIndex - i],
                                 "Shapes \(smaller) and \(larger) do not have common parts")
    }
    return TensorShape(larger.dropLast(smaller.count))
}
