// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#if !COMPILING_TENSORFLOW_MODULE
import TensorFlow
#endif

//===------------------------------------------------------------------------------------------===//
// Shape Transformations
//===------------------------------------------------------------------------------------------===//

public extension TensorFlowScalar {
    /// Convert to a tensor with the specified rank, with all dimensions equal to 1.
    @inlinable
    func makeTensor(rank: Int) -> Tensor<Self> {
        return Tensor(repeating: self, shape: TensorShape(rank))
    }
}

public extension Tensor {
    /// Reshape to the shape of the specified `Tensor`.
    /// - Precondition: The number of scalars matches the new shape.
    @inlinable
    @differentiable(wrt: self where Scalar : TensorFlowFloatingPoint)
    func reshaped<T>(like other: Tensor<T>) -> Tensor {
        return reshaped(toShape: other.shapeTensor)
    }

    /// Reshape to the specified shape.
    /// - Precondition: The number of scalars matches the new shape.
    @inlinable
    @differentiable(wrt: self where Scalar : TensorFlowFloatingPoint)
    func reshaped(to newShape: TensorShape) -> Tensor {
        // TODO(TF-433): Remove workaround for differentiating `map`.
        return reshaped(toShape: Tensor<Int32>({newShape.dimensions.map(Int32.init)}()))
    }

    /// Reshape to the specified `Tensor` representing a shape.
    /// - Precondition: The number of scalars matches the new shape.
    @inlinable
    @differentiable(
        wrt: self,
        vjp: _vjpReshaped(toShape:) where Scalar : TensorFlowFloatingPoint)
    func reshaped(toShape newShape: Tensor<Int32>) -> Tensor {
        return Raw.reshape(self, shape: newShape)
    }

    /// Return a copy of the tensor collapsed into a 1-D `Tensor`, in row-major order.
    @inlinable
    @differentiable(wrt: self where Scalar : TensorFlowFloatingPoint)
    func flattened() -> Tensor {
        return reshaped(to: [-1])
    }

    /// Returns a shape-expanded `Tensor`, with a dimension of 1 inserted at the
    /// specified shape index.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpExpandingShape(at:) where Scalar : TensorFlowFloatingPoint)
    func expandingShape(at shapeIndex: Int) -> Tensor {
        return Raw.expandDims(self, dim: Tensor<Int32>(Int32(shapeIndex)))
    }

    /// Returns a rank-lifted `Tensor` with a leading dimension of 1.
    @inlinable
    @differentiable(wrt: self where Scalar : TensorFlowFloatingPoint)
    func rankLifted() -> Tensor {
        return expandingShape(at: 0)
    }

    /// Remove the specified dimensions of size 1 from the shape of a tensor. If no dimensions are
    /// specified, then all dimensions of size 1 will be removed.
    @inlinable
    @differentiable(wrt: self where Scalar : TensorFlowFloatingPoint)
    func squeezingShape(at axes: Int...) -> Tensor {
        return squeezingShape(at: axes)
    }

    /// Remove the specified dimensions of size 1 from the shape of a tensor. If no dimensions are
    /// specified, then all dimensions of size 1 will be removed.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpSqueezingShape(at:) where Scalar : TensorFlowFloatingPoint)
    func squeezingShape(at axes: [Int]) -> Tensor {
        return Raw.squeeze(self, squeezeDims: axes.map(Int32.init))
    }
}

internal extension Tensor where Scalar : TensorFlowFloatingPoint {
    @inlinable
    func _vjpReshaped(toShape newShape: Tensor<Int32>) -> (Tensor, (Tensor) -> Tensor) {
        let value = reshaped(toShape: newShape)
        return (value, { [shape = shapeTensor] v in v.reshaped(toShape: shape) })
    }

    @inlinable
    func _vjpExpandingShape(at shapeIndex: Int) -> (Tensor, (Tensor) -> Tensor) {
        let value = expandingShape(at: shapeIndex)
        return (value, { v in v.squeezingShape(at: shapeIndex) })
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

infix operator ++ : AdditionPrecedence

public extension Tensor {
    /// Returns a transposed tensor, with dimensions permuted in the specified order.
    @inlinable
    @differentiable(
        wrt: self,
        vjp: _vjpTransposed(withPermutations:) where Scalar : TensorFlowFloatingPoint)
    func transposed(withPermutations permutations: Tensor<Int32>) -> Tensor {
        return Raw.transpose(self, perm: permutations)
    }

    /// Returns a transposed tensor, with dimensions permuted in the specified order.
    @inlinable
    @differentiable(
        wrt: self,
        vjp: _vjpTransposed(withPermutations:) where Scalar : TensorFlowFloatingPoint)
    func transposed(withPermutations permutations: [Int]) -> Tensor {
        let permutations = permutations.map(Int32.init)
        return transposed(withPermutations: Tensor<Int32>(permutations))
    }

    /// Returns a transposed tensor, with dimensions permuted in the specified order.
    @inlinable
    @differentiable(
        wrt: self, vjp: _vjpTransposed(withPermutations:) where Scalar : TensorFlowFloatingPoint)
    func transposed(withPermutations permutations: Int...) -> Tensor {
        return transposed(withPermutations: permutations)
    }

    /// Returns a transposed tensor, with dimensions permuted in reverse order.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpTransposed() where Scalar : TensorFlowFloatingPoint)
    func transposed() -> Tensor {
        let defaultPermutations = rankTensor - 1 - Tensor<Int32>(
            rangeFrom: 0, to: Int32(rank), stride: 1)
        return transposed(withPermutations: Tensor<Int32>(defaultPermutations))
    }

    /// Concatenates tensors along the specified axis.
    /// - Precondition: The tensors must have the same dimensions, except for the
    ///   specified axis.
    /// - Precondition: The axis must be in the range `-rank..<rank`.
    @inlinable
    @differentiable(vjp: _vjpConcatenated where Scalar : TensorFlowFloatingPoint)
    func concatenated(with other: Tensor, alongAxis axis: Int = 0) -> Tensor {
        return Tensor(concatenating: [self, other], alongAxis: axis)
    }

    /// Concatenation operator.
    /// - Note: `++` is a custom operator that does not exist in Swift, but does
    ///   in Haskell/Scala. Its addition is not an insignificant language change
    ///   and may be controversial. The existence/naming of `++` will be discussed
    ///   during a later API design phase.
    @inlinable
    @differentiable(where Scalar : TensorFlowFloatingPoint)
    static func ++ (lhs: Tensor, rhs: Tensor) -> Tensor {
        return lhs.concatenated(with: rhs)
    }
}

internal extension Tensor where Scalar : TensorFlowFloatingPoint {
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
        let idx = axis < 0 ? axis + rank : axis
        let splits = Tensor<Int32>([shapeTensor[idx], other.shapeTensor[idx]])
        return (concatenated(with: other, alongAxis: axis), { result in
            let gradients = Raw.splitV(
                value: result,
                sizeSplits: splits,
                splitDim: Tensor<Int32>(Int32(axis)),
                numSplit: Int64(2))
            return (gradients[0], gradients[1])
        })
    }
}

//===------------------------------------------------------------------------------------------===//
// Broadcasting
//===------------------------------------------------------------------------------------------===//

// TODO: What about precedence? Also, why is this operator meaningful for broadcasting?
infix operator .=

public extension Tensor {
    @inlinable
    func broadcast(toShape shape: Tensor<Int32>) -> Tensor {
        return Raw.broadcastTo(self, shape: shape)
    }

    @inlinable
    func broadcast(to shape: TensorShape) -> Tensor {
        return broadcast(toShape: Tensor<Int32>(shape.dimensions.map(Int32.init)))
    }

    /// Broadcast to the same shape as the specified `Tensor`.
    /// - Precondition: The specified shape must be compatible for broadcasting.
    @inlinable
    func broadcast<OtherScalar>(like other: Tensor<OtherScalar>) -> Tensor {
        return broadcast(toShape: other.shapeTensor)
    }

    @inlinable
    static func .= (lhs: inout Tensor, rhs: Tensor) {
        lhs = rhs.broadcast(like: lhs)
    }
}

// TODO: Why is this limited only to numeric data types whereas `broadcast` is not?
public extension Tensor where Scalar : Numeric {
    @inlinable
    func unbroadcast(toShape otherShape: Tensor<Int32>) -> Tensor {
        let rankDiff = (rankTensor - otherShape.scalarCountTensor).rankLifted()
        let ones: Tensor<Int32> = Raw.fill(dims: rankDiff, value: Tensor<Int32>(1))
        let paddedShape = ones ++ otherShape
        let nonEqualIndices = paddedShape .!= shapeTensor
        let broadcastIndices = Raw.where_(nonEqualIndices).flattened()
        let unbroadcasted: Tensor = Raw.sum(
            self, reductionIndices: Tensor<Int32>(broadcastIndices), keepDims: false)
        return Raw.reshape(unbroadcasted, shape: otherShape)
    }

    @inlinable
    func unbroadcast<OtherScalar>(like other: Tensor<OtherScalar>) -> Tensor {
        return unbroadcast(toShape: other.shapeTensor)
    }

    @inlinable
    func unbroadcast(to shape: TensorShape) -> Tensor {
        return unbroadcast(toShape: Tensor<Int32>(shape.dimensions.map(Int32.init)))
    }
}

//===------------------------------------------------------------------------------------------===//
// Padding
//===------------------------------------------------------------------------------------------===//

public extension Tensor where Scalar : Numeric {
    /// Returns a padded tensor according to the specified padding sizes.
    @inlinable
    func padded(forSizes sizes: [(before: Int, after: Int)], with value: Scalar = 0) -> Tensor {
        let paddings = Tensor<Int32>(
            shape: [sizes.count, 2],
            scalars: sizes.flatMap { [Int32($0.before), Int32($0.after)] })
        return Raw.padV2(self, paddings: paddings, constantValues: Tensor(value))
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
    @differentiable(wrt: self)
    func slice(lowerBounds: [Int], upperBounds: [Int]) -> Tensor {
        // TODO: Precondition `lowerBounds.count == upperBounds.count`,
        // preferably in graph.
        // TODO: Differentiating control flow is not supported yet, thus the thunks.
        let lowerBoundsTensor = Tensor<Int32>({lowerBounds.map(Int32.init)}())
        let upperBoundsTensor = Tensor<Int32>({upperBounds.map(Int32.init)}())
        return slice(lowerBounds: lowerBoundsTensor, sizes: upperBoundsTensor - lowerBoundsTensor)
    }

    @inlinable
    @differentiable(wrt: self, vjp: _vjpSlice)
    func slice(lowerBounds: Tensor<Int32>, sizes: Tensor<Int32>) -> Tensor {
        return Raw.slice(self, begin: lowerBounds, size: sizes)
    }

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

public enum TensorRange : TensorRangeExpression {
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

extension TensorRange : Equatable {
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
// extension UnboundedRange : TensorRangeExpression {
//     public var tensorRange: TensorRange { return .ellipsis }
// }

extension Int : TensorRangeExpression {
    public var tensorRange: TensorRange { return .index(self) }
}

extension Range : TensorRangeExpression where Bound == Int {
    public var tensorRange: TensorRange {
        return .range(self, stride: 1)
    }
}

extension ClosedRange : TensorRangeExpression where Bound == Int {
    public var tensorRange: TensorRange {
        return .closedRange(self, stride: 1)
    }
}

extension PartialRangeFrom : TensorRangeExpression where Bound == Int {
    public var tensorRange: TensorRange {
        return .partialRangeFrom(self, stride: 1)
    }
}

extension PartialRangeUpTo : TensorRangeExpression where Bound == Int {
    public var tensorRange: TensorRange {
        return .partialRangeUpTo(self, stride: 1)
    }
}

extension PartialRangeThrough : TensorRangeExpression where Bound == Int {
    public var tensorRange: TensorRange {
        return .partialRangeThrough(self, stride: 1)
    }
}

infix operator .. : StridedRangeFormationPrecedence
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
    @_fixed_layout @usableFromInline
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
    @differentiable(wrt: self, vjp: _vjpSubscript)
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
    // TODO: @differentiable(wrt: self)
    subscript(_ ranges: TensorRangeExpression...) -> Tensor {
        get {
            return self[IndexPath(ranges.map { $0.tensorRange })]
        }
        set {
            self[IndexPath(ranges.map { $0.tensorRange })] = newValue
        }
    }

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
        precondition(ranges.count { $0 == TensorRange.ellipsis } < 2,
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
