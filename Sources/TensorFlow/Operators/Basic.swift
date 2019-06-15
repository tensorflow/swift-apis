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

infix operator .!=: ComparisonPrecedence

/// Returns a tensor with the same shape and scalars as the specified tensor.
@inlinable
@differentiable(where Scalar: TensorFlowFloatingPoint)
public func identity<Scalar>(_ x: Tensor<Scalar>) -> Tensor<Scalar> {
    return x
}

//===------------------------------------------------------------------------------------------===//
// Shape Transformations
//===------------------------------------------------------------------------------------------===//

public extension TensorFlowScalar {
    /// Convert to a tensor with the specified rank, with all dimensions equal to `1`.
    @inlinable
    func makeTensor(rank: Int) -> Tensor<Self> {
        return Tensor(repeating: self, shape: TensorShape(rank))
    }
}

public extension Tensor {
    /// Unpacks the given dimension of a rank-`R` tensor into multiple rank-`(R-1)` tensors. Unpacks
    /// `N` tensors from this tensor by chipping it along the `axis` dimension, where `N` is
    /// inferred from this tensor's shape. For example, given a tensor with shape `[A, B, C, D]`:
    ///
    ///   - If `axis == 0` then the `i`-th tensor in the returned array is the slice
    ///     `self[i, :, :, :]` and each tensor in that array will have shape `[B, C, D]`.
    ///     (Note that the dimension unpacked along is gone, unlike
    ///     `Tensor.split(numSplits:alongAxis)`, or `Tensor.split(sizes:alongAxis)`).
    ///   - If `axis == 1` then the `i`-th tensor in the returned array is the slice
    ///     `value[:, i, :, :]` and each tensor in that array will have shape `[A, C, D]`.
    ///   - Etc.
    ///
    /// This is the opposite of `Tensor.init(stacking:alongAxis:)`.
    ///
    /// - Parameters:
    ///   - axis: Dimension along which to unstack. Negative values wrap around.
    ///
    /// - Precondition: `axis` must be in the range `[-rank, rank)`, where `rank` is the rank of the
    ///   provided tensors.
    ///
    /// - Returns: Array containing the unstacked tensors.
    @inlinable
    @differentiable(vjp: _vjpUnstacked(alongAxis:) where Scalar: TensorFlowFloatingPoint)
    func unstacked(alongAxis axis: Int = 0) -> [Tensor] {
        return Raw.unpack(value: self, num: Int64(shape[axis]), axis: Int64(axis))
    }

    /// Splits a tensor into multiple tensors. The tensor is split along dimension `axis` into
    /// `count` smaller tensors. This requires that `count` evenly divides `shape[axis]`.
    ///
    /// For example:
    /// ```
    /// // 'value' is a tensor with shape [5, 30]
    /// // Split 'value' into 3 tensors along dimension 1:
    /// let parts = value.split(count: 3, alongAxis: 1)
    /// parts[0] // has shape [5, 10]
    /// parts[1] // has shape [5, 10]
    /// parts[2] // has shape [5, 10]
    /// ```
    ///
    /// - Parameters:
    ///   - count: Number of splits to create.
    ///   - axis: The dimension along which to split this tensor. Negative values wrap around.
    ///
    /// - Precondition: `count` must divide the size of dimension `axis` evenly.
    /// - Precondition: `axis` must be in the range `[-rank, rank)`, where `rank` is the rank of the
    ///   provided tensors.
    ///
    /// - Returns: An array containing the tensors parts.
    @inlinable
    @differentiable(vjp: _vjpSplit(count:alongAxis:) where Scalar: TensorFlowFloatingPoint)
    func split(count: Int, alongAxis axis: Int = 0) -> [Tensor] {
        return Raw.split(
            splitDim: Tensor<Int32>(Int32(axis)), value: self, numSplit: Int64(count))
    }

    /// Splits a tensor into multiple tensors. The tensor is split  into `sizes.shape[0]` pieces.
    /// The shape of the `i`-th piece has the same shape as this tensor except along dimension
    /// `axis` where the size is `sizes[i]`.
    ///
    /// For example:
    /// ```
    /// // 'value' is a tensor with shape [5, 30]
    /// // Split 'value' into 3 tensors with sizes [4, 15, 11] along dimension 1:
    /// let parts = value.split(sizes: Tensor<Int32>([4, 15, 11]), alongAxis: 1)
    /// parts[0] // has shape [5, 4]
    /// parts[1] // has shape [5, 15]
    /// parts[2] // has shape [5, 11]
    /// ```
    ///
    /// - Parameters:
    ///   - sizes: 1-D tensor containing the size of each split.
    ///   - axis: Dimension along which to split this tensor. Negative values wrap around.
    ///
    /// - Precondition: The values in `sizes` must add up to the size of dimension `axis`.
    /// - Precondition: `axis` must be in the range `[-rank, rank)`, where `rank` is the rank of the
    ///   provided tensors.
    ///
    /// - Returns: Array containing the tensors parts.
    @inlinable
    @differentiable(
        wrt: self,
        vjp: _vjpSplit(sizes:alongAxis:) where Scalar: TensorFlowFloatingPoint)
    func split(sizes: Tensor<Int32>, alongAxis axis: Int = 0) -> [Tensor] {
        return Raw.splitV(
            value: self,
            sizeSplits: sizes,
            splitDim: Tensor<Int32>(Int32(axis)),
            numSplit: Int64(sizes.shape[0]))
    }

    /// Returns a tiled tensor, constructed by tiling this tensor.
    ///
    /// This constructor creates a new tensor by replicating this tensor `multiples` times. The
    /// constructed tensor's `i`'th dimension has `self.shape[i] * multiples[i]` elements, and the
    /// values of this tensor are replicated `multiples[i]` times along the `i`'th dimension. For
    /// example, tiling `[a b c d]` by `[2]` produces `[a b c d a b c d]`.
    ///
    /// - Precondition: The shape of `multiples` must be `[tensor.rank]`.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpTiled(multiples:) where Scalar: TensorFlowFloatingPoint)
    func tiled(multiples: Tensor<Int32>) -> Tensor {
        return Raw.tile(self, multiples: multiples)
    }

    /// Reshape to the shape of the specified `Tensor`.
    /// - Precondition: The number of scalars matches the new shape.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func reshaped<T>(like other: Tensor<T>) -> Tensor {
        return reshaped(toShape: other.shapeTensor)
    }

    /// Reshape to the specified shape.
    /// - Precondition: The number of scalars matches the new shape.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func reshaped(to newShape: TensorShape) -> Tensor {
        // TODO(TF-433): Remove workaround for differentiating `map`.
        return reshaped(toShape: Tensor<Int32>({newShape.dimensions.map(Int32.init)}()))
    }

    /// Reshape to the specified `Tensor` representing a shape.
    /// - Precondition: The number of scalars matches the new shape.
    @inlinable
    @differentiable(
        wrt: self,
        vjp: _vjpReshaped(toShape:) where Scalar: TensorFlowFloatingPoint)
    func reshaped(toShape newShape: Tensor<Int32>) -> Tensor {
        return Raw.reshape(self, shape: newShape)
    }

    /// Return a copy of the tensor collapsed into a 1-D `Tensor`, in row-major order.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func flattened() -> Tensor {
        return reshaped(to: [-1])
    }

    /// Returns a shape-expanded `Tensor`, with a dimension of 1 inserted at the specified shape
    /// indices.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func expandingShape(at axes: Int...) -> Tensor {
        return expandingShape(at: axes)
    }

    /// Returns a shape-expanded `Tensor`, with a dimension of 1 inserted at the
    /// specified shape indices.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpExpandingShape(at:) where Scalar: TensorFlowFloatingPoint)
    func expandingShape(at axes: [Int]) -> Tensor {
        var result = self
        for i in axes { result = Raw.expandDims(result, dim: Tensor<Int32>(Int32(i))) }
        return result
    }

    /// Returns a rank-lifted `Tensor` with a leading dimension of 1.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func rankLifted() -> Tensor {
        return expandingShape(at: 0)
    }

    /// Removes the specified dimensions of size 1 from the shape of a tensor. If no dimensions are
    /// specified, then all dimensions of size 1 will be removed.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func squeezingShape(at axes: Int...) -> Tensor {
        return squeezingShape(at: axes)
    }

    /// Removes the specified dimensions of size 1 from the shape of a tensor. If no dimensions are
    /// specified, then all dimensions of size 1 will be removed.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpSqueezingShape(at:) where Scalar: TensorFlowFloatingPoint)
    func squeezingShape(at axes: [Int]) -> Tensor {
        return Raw.squeeze(self, squeezeDims: axes.map(Int32.init))
    }
}

internal extension Tensor where Scalar: TensorFlowFloatingPoint {
    @inlinable
    func _vjpUnstacked(
        alongAxis axis: Int = 0
    ) -> ([Tensor], (Array<Tensor>.TangentVector) -> Tensor) {
        let result = unstacked(alongAxis: axis)
        return (result, { v in Tensor(stacking: v.base, alongAxis: axis) })
    }

    @inlinable
    func _vjpTiled(
        multiples: Tensor<Int32>
    ) -> (Tensor, (Tensor) -> Tensor) {
        return (tiled(multiples: multiples), { [shape = shapeTensor] v in
            let splitShape = Tensor<Int32>(stacking: [multiples, shape]).transposed().flattened()
            let axes = Tensor<Int32>(rangeFrom: 0, to: Int32(splitShape.scalarCount), stride: 2)
            return v.reshaped(toShape: splitShape).sum(squeezingAxes: axes)
        })
    }

    @inlinable
    func _vjpSplit(
        count: Int,
        alongAxis axis: Int = 0
    ) -> ([Tensor], (Array<Tensor>.TangentVector) -> Tensor) {
        let result = split(count: count, alongAxis: axis)
        return (result, { v in Tensor(concatenating: v.base, alongAxis: axis) })
    }

    @inlinable
    func _vjpSplit(
        sizes: Tensor<Int32>,
        alongAxis axis: Int = 0
    ) -> ([Tensor], (Array<Tensor>.TangentVector) -> Tensor) {
        let result = split(sizes: sizes, alongAxis: axis)
        return (result, { v in Tensor(concatenating: v.base, alongAxis: axis) })
    }

    @inlinable
    func _vjpReshaped(toShape newShape: Tensor<Int32>) -> (Tensor, (Tensor) -> Tensor) {
        let value = reshaped(toShape: newShape)
        return (value, { [shape = shapeTensor] v in v.reshaped(toShape: shape) })
    }

    @inlinable
    func _vjpExpandingShape(at axes: [Int]) -> (Tensor, (Tensor) -> Tensor) {
        let value = self.expandingShape(at: axes)
        return (value, { v in v.squeezingShape(at: axes) })
    }

    @inlinable
    func _vjpSqueezingShape(at axes: [Int]) -> (Tensor, (Tensor) -> Tensor) {
        let value = squeezingShape(at: axes)
        return (value, { [shape = shapeTensor] v in v.reshaped(toShape: shape) })
    }
}

//===------------------------------------------------------------------------------------------===//
// Other Tensor Transformations
//===------------------------------------------------------------------------------------------===//

infix operator ++: AdditionPrecedence

public extension Tensor {
    /// Returns a transposed tensor, with dimensions permuted in the specified order.
    @inlinable
    @differentiable(
        wrt: self,
        vjp: _vjpTransposed(withPermutations:) where Scalar: TensorFlowFloatingPoint)
    func transposed(withPermutations permutations: Tensor<Int32>) -> Tensor {
        return Raw.transpose(self, perm: permutations)
    }

    /// Returns a transposed tensor, with dimensions permuted in the specified order.
    @inlinable
    @differentiable(
        wrt: self,
        vjp: _vjpTransposed(withPermutations:) where Scalar: TensorFlowFloatingPoint)
    func transposed(withPermutations permutations: [Int]) -> Tensor {
        let permutations = permutations.map(Int32.init)
        return transposed(withPermutations: Tensor<Int32>(permutations))
    }

    /// Returns a transposed tensor, with dimensions permuted in the specified order.
    @inlinable
    @differentiable(
        wrt: self, vjp: _vjpTransposed(withPermutations:) where Scalar: TensorFlowFloatingPoint)
    func transposed(withPermutations permutations: Int...) -> Tensor {
        return transposed(withPermutations: permutations)
    }

    /// Returns a transposed tensor, with dimensions permuted in reverse order.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpTransposed() where Scalar: TensorFlowFloatingPoint)
    func transposed() -> Tensor {
        let defaultPermutations = rankTensor - 1 - Tensor<Int32>(
            rangeFrom: 0, to: Int32(rank), stride: 1)
        return transposed(withPermutations: Tensor<Int32>(defaultPermutations))
    }

    /// Returns a concatenated tensor along the specified axis.
    /// - Precondition: The tensors must have the same dimensions, except for the
    ///   specified axis.
    /// - Precondition: The axis must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(vjp: _vjpConcatenated where Scalar: TensorFlowFloatingPoint)
    func concatenated(with other: Tensor, alongAxis axis: Int = 0) -> Tensor {
        return Tensor(concatenating: [self, other], alongAxis: axis)
    }

    /// Concatenation operator.
    /// - Note: `++` is a custom operator that does not exist in Swift, but does
    ///   in Haskell/Scala. Its addition is not an insignificant language change
    ///   and may be controversial. The existence/naming of `++` will be discussed
    ///   during a later API design phase.
    @inlinable
    @differentiable(where Scalar: TensorFlowFloatingPoint)
    static func ++ (lhs: Tensor, rhs: Tensor) -> Tensor {
        return lhs.concatenated(with: rhs)
    }

    /// Returns a tensor by gathering slices of the input at `indices` along the `axis` dimension
    ///
    /// For 0-D (scalar) `indices`:
    /// ```
    /// result[p_0,          ..., p_{axis-1},
    ///        p_{axis + 1}, ..., p_{N-1}] =
    /// self[p_0,          ..., p_{axis-1},
    ///      indices,
    ///      p_{axis + 1}, ..., p_{N-1}]
    /// ```
    ///
    /// For 1-D (vector) `indices`:
    /// ```
    /// result[p_0,          ..., p_{axis-1},
    ///        i,
    ///        p_{axis + 1}, ..., p_{N-1}] =
    /// self[p_0,          ..., p_{axis-1},
    ///      indices[i],
    ///      p_{axis + 1}, ..., p_{N-1}]
    /// ```
    ///
    /// In the general case, produces a resulting tensor where:
    /// ```
    /// result[p_0,             ..., p_{axis-1},
    ///        i_{batch\_dims}, ..., i_{M-1},
    ///        p_{axis + 1},    ..., p_{N-1}] =
    /// self[p_0,             ..., p_{axis-1},
    ///      indices[i_0,     ..., i_{M-1}],
    ///      p_{axis + 1},    ..., p_{N-1}]
    /// ```
    /// where `N = self.rank` and `M = indices.rank`.
    ///
    /// The shape of the resulting tensor is:
    /// `self.shape[..<axis] + indices.shape + self.shape[(axis + 1)...]`.
    ///
    /// - Note: On CPU, if an out-of-range index is found, an error is thrown. On GPU, if an
    /// out-of-range index is found, a 0 is stored in the corresponding output values.
    ///
    /// - Parameters:
    ///   - indices: Contains the indices to gather at.
    ///   - axis: Dimension along which to gather. Negative values wrap around.
    ///
    /// - Precondition: `axis` must be in the range `[-rank, rank)`.
    ///
    /// - Returns: The gathered tensor.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpGathering where Scalar : TensorFlowFloatingPoint)
    func gathering(atIndices indices: Tensor<Int32>, alongAxis axis: Int = 0) -> Tensor {
        return Raw.gatherV2(params: self, indices: indices, axis: Tensor<Int32>(Int32(axis)))
    }

    /// Returns a tensor by gathering the values after applying the provided boolean mask to the input.
    ///
    /// For example:
    /// ```
    /// // 1-D example
    /// // tensor is [0, 1, 2, 3]
    /// // mask is [true, false, true, false]
    /// tensor.gathering(where: mask) // is [0, 2]
    ///
    /// // 2-D example
    /// // tensor is [[1, 2], [3, 4], [5, 6]]
    /// // mask is [true, false, true]
    /// tensor.gathering(where: mask) // is [[1, 2], [5, 6]]
    /// ```
    ///
    /// In general, `0 < mask.rank = K <= tensor.rank`, and the `mask`'s shape must match the first
    /// K dimensions of the `tensor`'s shape. We then have:
    /// `tensor.gathering(where: mask)[i, j1, ..., jd] = tensor[i1, ..., iK, j1, ..., jd]`, where
    /// `[i1, ..., iK]` is the `i`th `true` entry of `mask` (row-major order).
    ///
    /// The `axis` could be used with `mask` to indicate the axis to mask from. In that case,
    /// `axis + mask.rank <= tensor.rank` and the `mask``'s shape must match the first
    /// `axis + mask.rank` dimensions of the `tensor`'s shape.
    ///
    /// - Parameters:
    ///   - mask: K-D boolean tensor, where `K <= self.rank`.
    ///   - axis: 0-D integer tensor representing the axis in `self` to mask from, where
    ///     `K + axis <= self.rank`.
    ///
    /// - Precondition: The `mask` cannot be a scalar: `mask.rank != 0`.
    ///
    /// - Returns: `(self.rank - K + 1)`-dimensional tensor populated by entries in this tensor
    ///   corresponding to `true` values in `mask`.
    @inlinable
    // @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func gathering(where mask: Tensor<Bool>, alongAxis axis: Int = 0) -> Tensor {
        precondition(mask.rank != 0, "The boolean mask cannot be a scalar.")
        // TODO: Remove once control flow AD is supported.
        let rank = self.rank
        let posAxis = { axis < 0 ? axis + rank : axis }()
        let leadingSize = shapeTensor[posAxis ..< posAxis + mask.rank].product().rankLifted()
        let reshapedTensor = reshaped(
            toShape: Tensor<Int32>(concatenating: [
                shapeTensor[..<posAxis],
                leadingSize,
                shapeTensor[(posAxis + mask.rank)...]]))
        let indices = Tensor<Int32>(mask.flattened().nonZeroIndices().squeezingShape(at: 1))
        return reshapedTensor.gathering(atIndices: indices, alongAxis: posAxis)
    }
}

internal extension Tensor where Scalar: TensorFlowFloatingPoint {
    @inlinable
    func _vjpTransposed(
        withPermutations permutations: Tensor<Int32>
    ) -> (Tensor, (Tensor) -> Tensor) {
        let value = transposed(withPermutations: permutations)
        return (value, { $0.transposed(withPermutations: permutations) })
    }

    @inlinable
    func _vjpTransposed(withPermutations permutations: [Int]) -> (Tensor, (Tensor) -> Tensor) {
        let value = transposed(withPermutations: permutations)
        return (value, { $0.transposed(withPermutations: permutations) })
    }

    @inlinable
    func _vjpTransposed(withPermutations permutations: Int...) -> (Tensor, (Tensor) -> Tensor) {
        let value = transposed(withPermutations: permutations)
        return (value, { $0.transposed(withPermutations: permutations) })
    }

    @inlinable
    func _vjpTransposed() -> (Tensor, (Tensor) -> Tensor) {
        return (transposed(), { $0.transposed() })
    }

    @inlinable
    func _vjpConcatenated(
        with other: Tensor,
        alongAxis axis: Int
    ) -> (Tensor, (Tensor) -> (Tensor, Tensor)) {
        let posAxis = axis < 0 ? axis + rank: axis
        let splits = Tensor<Int32>([shapeTensor[posAxis], other.shapeTensor[posAxis]])
        return (concatenated(with: other, alongAxis: axis), { result in
            let gradients = result.split(sizes: splits, alongAxis: axis)
            return (gradients[0], gradients[1])
        })
    }

    @inlinable
    func _vjpGathering(
        atIndices indices: Tensor<Int32>,
        alongAxis axis: Int = 0
    ) -> (Tensor, (Tensor) -> Tensor) {
        let result = gathering(atIndices: indices, alongAxis: axis)
        let posAxis = axis < 0 ? axis + rank : axis

        // We have a fast gradient implementation for the case when `posAxis == 0`.
        if posAxis == 0 {
            return (result, { [shape = shapeTensor] v in
                let indicesCount = indices.scalarCountTensor.rankLifted()
                let valuesShape = Tensor<Int32>(concatenating: [indicesCount, shape[1...]])
                let values = v.reshaped(toShape: valuesShape)
                let valueIndices = indices.reshaped(toShape: indicesCount)
                return Raw.unsortedSegmentSum(
                    data: values,
                    segmentIds: valueIndices,
                    numSegments: shape[0])
            })
        }

        return (result, { [shape = shapeTensor] v in
            let indicesSize = Tensor<Int32>(Int32(indices.scalarCount)).rankLifted()
            let outerShape = shape[..<posAxis]
            let outerSize = outerShape.scalarCount
            let innerShape = shape[(posAxis + 1)...]
            let innerSize = innerShape.scalarCount
            let outerIndices = Tensor<Int32>(rangeFrom: 0, to: Int32(outerSize), stride: 1)
            let innerIndices = Tensor<Int32>(
                rangeFrom: Int32(outerSize) + 1,
                to: Int32(outerSize) + 1 + Int32(innerSize),
                stride: 1)
            let valuesShape = Tensor<Int32>(concatenating: [outerShape, indicesSize, innerShape])
            let values = v.reshaped(toShape: valuesShape)
            let valueIndices = indices.reshaped(toShape: indicesSize)

            // We need to sum up every slice `values[..., i, ....]` corresponding to
            // `tensor[..., indices[i], ...]`. Since `unsortedSegmentSum` does not support an axis
            // parameter, we transpose the gather dimension to the front, then use
            // `unsortedSegmentSum` to build a `[gatherAxis, outerAxes, innerAxes]` tensor with all
            // the gradients affecting each index in `gatherAxis` summed up.
            let permutations = Tensor<Int32>(concatenating: [
                Tensor<Int32>([Int32(outerSize)]),
                outerIndices,
                innerIndices])
            let transposedValues = values.transposed(withPermutations: permutations)
            let gradient = Raw.unsortedSegmentSum(
                data: transposedValues,
                segmentIds: valueIndices,
                numSegments: shape[posAxis])

            // Finally, we invert the above transpose operation by moving dimension 0 back to its
            // original position.
            let inversePermutations = Tensor<Int32>(concatenating: [
                outerIndices + 1,
                Tensor<Int32>([0]),
                innerIndices])
            return gradient.transposed(withPermutations: inversePermutations)
        })
    }
}

public extension Tensor {
    /// Returns the locations of non-zero / true values in this tensor.
    ///
    /// The coordinates are returned in a 2-D tensor where the first dimension (rows) represents the
    /// number of non-zero elements, and the second dimension (columns) represents the coordinates
    /// of the non-zero elements. Keep in mind that the shape of the output tensor can vary
    /// depending on how many true values there are in this tensor. Indices are output in row-major
    /// order.
    ///
    /// For example:
    /// ```
    /// // 'input' is [[true, false], [true, false]]
    /// // 'input' has 2 true values and so the output has 2 rows.
    /// // 'input' has rank of 2, and so the second dimension of the output has size 2.
    /// input.nonZeroIndices() // is [[0, 0], [1, 0]]
    ///
    /// // 'input' is [[[ true, false], [ true, false]],
    /// //             [[false,  true], [false,  true]],
    /// //             [[false, false], [false,  true]]]
    /// // 'input' has 5 true values and so the output has 5 rows.
    /// // 'input' has rank 3, and so the second dimension of the output has size 3.
    /// input.nonZeroIndices() // is [[0, 0, 0],
    ///                        //     [0, 1, 0],
    ///                        //     [1, 0, 1],
    ///                        //     [1, 1, 1],
    ///                        //     [2, 1, 1]]
    /// ```
    ///
    /// - Returns: A tensor with shape `(num_true, rank(condition))`.
    @inlinable
    func nonZeroIndices() -> Tensor<Int64> {
        return Raw.where_(self)
    }
}

//===------------------------------------------------------------------------------------------===//
// Broadcasting
//===------------------------------------------------------------------------------------------===//

// TODO: What about precedence? Why is this operator used for broadcasting?
infix operator .=

public extension Tensor {
    @inlinable
    @differentiable(wrt: self, vjp: _vjpBroadcasted(toShape:) where Scalar: TensorFlowFloatingPoint)
    func broadcasted(toShape shape: Tensor<Int32>) -> Tensor {
        return Raw.broadcastTo(self, shape: shape)
    }

    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func broadcasted(to shape: TensorShape) -> Tensor {
        return broadcasted(toShape: Tensor<Int32>({ shape.dimensions.map(Int32.init) }()))
    }

    /// Broadcast to the same shape as the specified `Tensor`.
    /// - Precondition: The specified shape must be compatible for broadcasting.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func broadcasted<OtherScalar>(like other: Tensor<OtherScalar>) -> Tensor {
        return broadcasted(toShape: other.shapeTensor)
    }

    @inlinable
    static func .= (lhs: inout Tensor, rhs: Tensor) {
        lhs = rhs.broadcasted(like: lhs)
    }
}

public extension Tensor where Scalar: Numeric {
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func unbroadcasted(toShape otherShape: Tensor<Int32>) -> Tensor {
        // TODO: Simplify this once differentiating control flow is supported.
	    return unbroadcasted(to: {
	      precondition(otherShape.rank == 1)
	      return TensorShape(otherShape.scalars.map(Int.init))
	    }())
    }

    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func unbroadcasted<OtherScalar>(like other: Tensor<OtherScalar>) -> Tensor {
        return unbroadcasted(toShape: other.shapeTensor)
    }

    @inlinable
    @differentiable(wrt: self, vjp: _vjpUnbroadcasted(to:) where Scalar: TensorFlowFloatingPoint)
    func unbroadcasted(to shape: TensorShape) -> Tensor {
        let dimensions = self.shape.dimensions
	    var otherDimensions = shape.dimensions
	    let rankDifference = dimensions.count - otherDimensions.count
	    precondition(rankDifference >= 0, """
	        The rank of 'self' must be greater than or equal to the number of \
	        dimensions in the destination shape
	        """)
	    if rankDifference > 0 {
            otherDimensions.insert(contentsOf: repeatElement(1, count: rankDifference), at: 0)
	    }
	    assert(dimensions.count == otherDimensions.count)
	    var axes: [Int] = []
	    axes.reserveCapacity(dimensions.count)
	    for (i, (dim, otherDim)) in zip(dimensions, otherDimensions).enumerated() {
	        if dim == otherDim { continue }
	        if otherDim == 1 { axes.append(i); continue }
	        preconditionFailure("Cannot unbroadcast \(self.shape) to \(shape)")
	    }
	    return sum(alongAxes: axes).reshaped(to: shape)
    }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
    @inlinable
    func _vjpBroadcasted(toShape shape: Tensor<Int32>) -> (Tensor, (Tensor) -> Tensor) {
        return (broadcasted(toShape: shape), { [originalShape = shapeTensor] v in
            v.unbroadcasted(toShape: originalShape)
        })
    }

    @inlinable
    func _vjpUnbroadcasted(to shape: TensorShape) -> (Tensor, (Tensor) -> Tensor) {
        return (unbroadcasted(to: shape), { [originalShape = shapeTensor] v in
            v.broadcasted(toShape: originalShape)
        })
    }
}

//===------------------------------------------------------------------------------------------===//
// Padding
//===------------------------------------------------------------------------------------------===//

public extension Tensor where Scalar: Numeric {
    /// Returns a padded tensor according to the specified padding sizes.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpPadded(forSizes:with:) where Scalar: TensorFlowFloatingPoint)
    func padded(forSizes sizes: [(before: Int, after: Int)], with value: Scalar = 0) -> Tensor {
        let paddings = Tensor<Int32>(
            shape: [sizes.count, 2],
            scalars: sizes.flatMap { [Int32($0.before), Int32($0.after)] })
        return Raw.padV2(self, paddings: paddings, constantValues: Tensor(value))
    }
}

internal extension Tensor where Scalar: TensorFlowFloatingPoint {
    @inlinable
    func _vjpPadded(
        forSizes sizes: [(before: Int, after: Int)],
        with value: Scalar
    ) -> (Tensor, (Tensor) -> Tensor) {
        let result = padded(forSizes: sizes, with: value)
        return (result, { [rank = rankTensor, shape = shapeTensor] v in
            let paddings = Tensor<Int32>(
                shape: [sizes.count, 2],
                scalars: sizes.flatMap { [Int32($0.before), Int32($0.after)] })
            let padBefore = Raw.slice(paddings,
                begin: Tensor<Int32>([0, 0]),
                size: Tensor<Int32>(stacking: [rank, Tensor<Int32>(1)]))
            let begin = padBefore.reshaped(to: [-1])
            return Raw.slice(v, begin: begin, size: shape)
        })
    }
}

//===------------------------------------------------------------------------------------------===//
// Indexing and Slicing
//===------------------------------------------------------------------------------------------===//

// TODO: Negative indexing and strides syntax.

public extension Tensor {
    /// Extracts a slice from the tensor defined by lower and upper bounds for
    /// each dimension.
    ///
    /// - Parameter lowerBounds: The lower bounds at each dimension.
    /// - Parameter upperBounds: The upper bounds at each dimension.
    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    func slice(lowerBounds: [Int], upperBounds: [Int]) -> Tensor {
        // TODO: Precondition `lowerBounds.count == upperBounds.count`,
        // preferably in graph.
        // TODO: Differentiating control flow is not supported yet, thus the thunks.
        let lowerBoundsTensor = Tensor<Int32>({lowerBounds.map(Int32.init)}())
        let upperBoundsTensor = Tensor<Int32>({upperBounds.map(Int32.init)}())
        return slice(lowerBounds: lowerBoundsTensor, sizes: upperBoundsTensor - lowerBoundsTensor)
    }

    @inlinable
    @differentiable(wrt: self, vjp: _vjpSlice where Scalar: TensorFlowFloatingPoint)
    func slice(lowerBounds: Tensor<Int32>, sizes: Tensor<Int32>) -> Tensor {
        return Raw.slice(self, begin: lowerBounds, size: sizes)
    }
}

public extension Tensor where Scalar: TensorFlowFloatingPoint {
    @inlinable
    internal func _vjpSlice(
        lowerBounds: Tensor<Int32>,
        sizes: Tensor<Int32>
    ) -> (Tensor, (Tensor) -> Tensor) {
        let value = slice(lowerBounds: lowerBounds, sizes: sizes)
        let afterPaddings = shapeTensor - value.shapeTensor - lowerBounds
        return (value, { [after = afterPaddings] v in
            let beforePaddings = lowerBounds.expandingShape(at: 1)
            let afterPaddings = after.expandingShape(at: 1)
            let paddings = Tensor<Int32>(
                concatenating: [beforePaddings, afterPaddings], alongAxis: 1)
            return Raw.pad(v, paddings: paddings)
        })
    }
}

public enum TensorRange: TensorRangeExpression {
    case ellipsis
    case newAxis
    case squeezeAxis
    case index(Int)
    case range(Range<Int>, stride: Int)
    case closedRange(ClosedRange<Int>, stride: Int)
    case partialRangeFrom(PartialRangeFrom<Int>, stride: Int)
    case partialRangeUpTo(PartialRangeUpTo<Int>, stride: Int)
    case partialRangeThrough(PartialRangeThrough<Int>, stride: Int)

    public var tensorRange: TensorRange { return self }
}

extension TensorRange: Equatable {
    public static func == (lhs: TensorRange, rhs: TensorRange) -> Bool {
        switch (lhs, rhs) {
        case (.ellipsis, .ellipsis),
             (.newAxis, .newAxis),
             (.squeezeAxis, .squeezeAxis):
            return true
        case (let .index(i1), let .index(i2)): return i1 == i2
        case (let .range(r1, s1), let .range(r2, s2)): return r1 == r2 && s1 == s2
        case (let .closedRange(r1, s1), let .closedRange(r2, s2)):
            return r1 == r2 && s1 == s2
        case (let .partialRangeFrom(r1, s1), let .partialRangeFrom(r2, s2)):
            return r1.lowerBound == r2.lowerBound && s1 == s2
        case (let .partialRangeUpTo(r1, s1), let .partialRangeUpTo(r2, s2)):
            return r1.upperBound == r2.upperBound && s1 == s2
        case (let .partialRangeThrough(r1, s1), let .partialRangeThrough(r2, s2)):
            return r1.upperBound == r2.upperBound && s1 == s2
        default: return false
        }
    }
}

public protocol TensorRangeExpression {
    var tensorRange: TensorRange { get }
}

// TODO: Cannot extend non-nominal type 'UnboundedRange'.
// extension UnboundedRange: TensorRangeExpression {
//     public var tensorRange: TensorRange { return .ellipsis }
// }

extension Int: TensorRangeExpression {
    public var tensorRange: TensorRange { return .index(self) }
}

extension Range: TensorRangeExpression where Bound == Int {
    public var tensorRange: TensorRange {
        return .range(self, stride: 1)
    }
}

extension ClosedRange: TensorRangeExpression where Bound == Int {
    public var tensorRange: TensorRange {
        return .closedRange(self, stride: 1)
    }
}

extension PartialRangeFrom: TensorRangeExpression where Bound == Int {
    public var tensorRange: TensorRange {
        return .partialRangeFrom(self, stride: 1)
    }
}

extension PartialRangeUpTo: TensorRangeExpression where Bound == Int {
    public var tensorRange: TensorRange {
        return .partialRangeUpTo(self, stride: 1)
    }
}

extension PartialRangeThrough: TensorRangeExpression where Bound == Int {
    public var tensorRange: TensorRange {
        return .partialRangeThrough(self, stride: 1)
    }
}

infix operator ..: StridedRangeFormationPrecedence
precedencegroup StridedRangeFormationPrecedence {
    associativity: left
    higherThan: CastingPrecedence
    lowerThan: RangeFormationPrecedence
}

public extension Range where Bound == Int {
    static func .. (range: Range, stride: Int) -> TensorRange {
        return .range(range, stride: stride)
    }
}

public extension ClosedRange where Bound == Int {
    static func .. (range: ClosedRange, stride: Int) -> TensorRange {
        return .closedRange(range, stride: stride)
    }
}

public extension PartialRangeFrom where Bound == Int {
    static func .. (range: PartialRangeFrom, stride: Int) -> TensorRange {
        return .partialRangeFrom(range, stride: stride)
    }
}

public extension PartialRangeUpTo where Bound == Int {
    static func .. (range: PartialRangeUpTo, stride: Int) -> TensorRange {
        return .partialRangeUpTo(range, stride: stride)
    }
}

public extension PartialRangeThrough where Bound == Int {
    static func .. (range: PartialRangeThrough, stride: Int) -> TensorRange {
        return .partialRangeThrough(range, stride: stride)
    }
}

public extension Tensor {
    @frozen @usableFromInline
    internal struct IndexPath {
        @usableFromInline
        let begin, end, strides: Tensor<Int32>

        @usableFromInline
        let beginMask, endMask, ellipsisMask, newAxisMask, squeezeAxisMask: Int64

        @inlinable
        public init(
            begin: Tensor<Int32>, end: Tensor<Int32>, strides: Tensor<Int32>,
            beginMask: Int64, endMask: Int64, ellipsisMask: Int64, newAxisMask: Int64,
            squeezeAxisMask: Int64
        ) {
            self.begin = begin
            self.end = end
            self.strides = strides
            self.beginMask = beginMask
            self.endMask = endMask
            self.ellipsisMask = ellipsisMask
            self.newAxisMask = newAxisMask
            self.squeezeAxisMask = squeezeAxisMask
        }
    }

    @inlinable
    @differentiable(wrt: self, vjp: _vjpSubscript where Scalar : TensorFlowFloatingPoint)
    internal subscript(_ indexPath: IndexPath) -> Tensor {
        get {
            return Raw.stridedSlice(
                self, begin: indexPath.begin, end: indexPath.end,
                strides: indexPath.strides, beginMask: indexPath.beginMask,
                endMask: indexPath.endMask, ellipsisMask: indexPath.ellipsisMask,
                newAxisMask: indexPath.newAxisMask,
                shrinkAxisMask: indexPath.squeezeAxisMask)
        }
        set {
            self = Raw.tensorStridedSliceUpdate(
                self, begin: indexPath.begin, end: indexPath.end,
                strides: indexPath.strides, value: newValue,
                beginMask: indexPath.beginMask, endMask: indexPath.endMask,
                ellipsisMask: indexPath.ellipsisMask,
                newAxisMask: indexPath.newAxisMask,
                shrinkAxisMask: indexPath.squeezeAxisMask)
        }
    }

    @inlinable
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    subscript(_ ranges: TensorRangeExpression...) -> Tensor {
        get {
            return self[{IndexPath({ranges.map { $0.tensorRange }}())}()]
        }
        set {
            self[{IndexPath({ranges.map { $0.tensorRange }}())}()] = newValue
        }
    }
}

public extension Tensor where Scalar: TensorFlowFloatingPoint {
    @usableFromInline
    internal func _vjpSubscript(
        _ indexPath: IndexPath
    ) -> (Tensor, (Tensor) -> Tensor) {
        return (self[indexPath], { [shape = shapeTensor] v in
            Raw.stridedSliceGrad(
                shape: shape, begin: indexPath.begin, end: indexPath.end,
                strides: indexPath.strides, dy: v, beginMask: indexPath.beginMask,
                endMask: indexPath.endMask, ellipsisMask: indexPath.ellipsisMask,
                newAxisMask: indexPath.newAxisMask,
                shrinkAxisMask: indexPath.squeezeAxisMask)
        })
    }
}

internal extension Tensor.IndexPath {
    @inlinable
    init(_ ranges: [TensorRange]) {
        precondition(!ranges.isEmpty, "The tensor range collection cannot be empty.")
        precondition(ranges.lazy.filter { $0 == TensorRange.ellipsis }.count < 2,
                     "Only one ellipsis is allowed per tensor range collection.")

        var begin = [Int32](repeating: 0, count: ranges.count)
        var end = [Int32](repeating: 0, count: ranges.count)
        var strides = [Int32](repeating: 1, count: ranges.count)
        var beginMask: Int64 = 0
        var endMask: Int64 = 0
        var ellipsisMask: Int64 = 0
        var newAxisMask: Int64 = 0
        var squeezeAxisMask: Int64 = 0
        for (i, index) in ranges.enumerated() {
            switch index {
            case .ellipsis: ellipsisMask |= 1 << i
            case .newAxis: newAxisMask |= 1 << i
            case .squeezeAxis: squeezeAxisMask |= 1 << i
            case .index(let index):
                begin[i] = Int32(index)
                end[i] = Int32(index) + 1
                squeezeAxisMask |= 1 << i
            case .range(let range, let stride):
                begin[i] = Int32(range.lowerBound)
                end[i] = Int32(range.upperBound)
                strides[i] = Int32(stride)
            case .closedRange(let range, let stride):
                begin[i] = Int32(range.lowerBound)
                switch Int32(range.upperBound) {
                case -1: endMask |= 1 << i
                case let u: end[i] = u + 1
                }
                strides[i] = Int32(stride)
            case .partialRangeFrom(let range, let stride):
                begin[i] = Int32(range.lowerBound)
                strides[i] = Int32(stride)
                endMask |= 1 << i
            case .partialRangeUpTo(let range, let stride):
                end[i] = Int32(range.upperBound)
                strides[i] = Int32(stride)
                beginMask |= 1 << i
            case .partialRangeThrough(let range, let stride):
                end[i] = Int32(range.upperBound) + 1
                strides[i] = Int32(stride)
                beginMask |= 1 << i
            }
        }

        self.begin = Tensor<Int32>(begin)
        self.end = Tensor<Int32>(end)
        self.strides = Tensor<Int32>(strides)
        self.beginMask = beginMask
        self.endMask = endMask
        self.ellipsisMask = ellipsisMask
        self.newAxisMask = newAxisMask
        self.squeezeAxisMask = squeezeAxisMask
    }
}
