
import _Differentiation

infix operator .!=: ComparisonPrecedence

@inlinable
@differentiable(where Scalar: TensorFlowFloatingPoint)
public func identity<Scalar>(_ x: Tensor<Scalar>) -> Tensor<Scalar> {
return x
}


extension TensorFlowScalar {
  @inlinable
  public func makeTensor(rank: Int, on device: Device = .default) -> Tensor<Self> {
    fatalError()
  }
}

extension Tensor {
  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public func unstacked(alongAxis axis: Int = 0) -> [Tensor] {
    fatalError()
  }

  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public func split(count: Int, alongAxis axis: Int = 0) -> [Tensor] {
    fatalError()
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func split(sizes: Tensor<Int32>, alongAxis axis: Int = 0) -> [Tensor] {
    fatalError()
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func split(sizes: [Int], alongAxis axis: Int = 0) -> [Tensor] {
    fatalError()
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func tiled(multiples: [Int]) -> Tensor {
    fatalError()
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func tiled(multiples: Tensor<Int32>) -> Tensor {
    fatalError()
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func reshaped<T>(like other: Tensor<T>) -> Tensor {
  return self
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func reshaped(to newShape: TensorShape) -> Tensor {
    fatalError()
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func reshaped(toShape newShape: Tensor<Int32>) -> Tensor {
    fatalError()
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func flattened() -> Tensor {
  return self
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func expandingShape(at axes: Int...) -> Tensor {
  return self
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func expandingShape(at axes: [Int]) -> Tensor {
    fatalError()
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func rankLifted() -> Tensor {
  return self
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func squeezingShape(at axes: Int...) -> Tensor {
  return self
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func squeezingShape(at axes: [Int]) -> Tensor {
    fatalError()
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: unstacked)
  func _vjpUnstacked(
    alongAxis axis: Int = 0
  ) -> (value: [Tensor], pullback: (Array<Tensor>.TangentVector) -> Tensor) {
    fatalError()
  }

  @inlinable
  @derivative(of: tiled)
  func _vjpTiled(multiples: Tensor<Int32>) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    fatalError()
  }

  @inlinable
  @derivative(of: tiled)
  func _vjpTiled(multiples: [Int]) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    fatalError()
  }

  @inlinable
  @derivative(of: split)
  func _vjpSplit(
    count: Int,
    alongAxis axis: Int = 0
  ) -> (value: [Tensor], pullback: (Array<Tensor>.TangentVector) -> Tensor) {
    fatalError()
  }

  @inlinable
  @derivative(of: split)
  func _vjpSplit(
    sizes: Tensor<Int32>,
    alongAxis axis: Int = 0
  ) -> (value: [Tensor], pullback: (Array<Tensor>.TangentVector) -> Tensor) {
    fatalError()
  }

  @inlinable
  @derivative(of: split)
  func _vjpSplit(
    sizes: [Int],
    alongAxis axis: Int = 0
  ) -> (value: [Tensor], pullback: (Array<Tensor>.TangentVector) -> Tensor) {
    fatalError()
  }

  @inlinable
  @derivative(of: reshaped)
  func _vjpReshaped(toShape newShape: Tensor<Int32>) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    fatalError()
  }

  @inlinable
  @derivative(of: reshaped)
  func _vjpReshaped(toShape newShape: TensorShape) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    fatalError()
  }

  @inlinable
  @derivative(of: expandingShape)
  func _vjpExpandingShape(at axes: [Int]) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    fatalError()
  }

  @inlinable
  @derivative(of: squeezingShape)
  func _vjpSqueezingShape(at axes: [Int]) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    fatalError()
  }
}


infix operator ++: AdditionPrecedence

extension Tensor {
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func transposed(permutation: Tensor<Int32>) -> Tensor {
    fatalError()
  }

  @available(*, deprecated, renamed: "transposed(permutation:)")
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func transposed(withPermutations permutations: Tensor<Int32>) -> Tensor {
  return self
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func transposed(permutation: [Int]) -> Tensor {
    fatalError()
  }

  @available(*, deprecated, renamed: "transposed(permutation:)")
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func transposed(withPermutations permutations: [Int]) -> Tensor {
  return self
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func transposed(permutation: Int...) -> Tensor {
    fatalError()
  }

  @available(*, deprecated, renamed: "transposed(permutation:)")
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func transposed(withPermutations permutations: Int...) -> Tensor {
  return self
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func transposed() -> Tensor {
    fatalError()
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func reversed(inAxes axes: Tensor<Int32>) -> Tensor {
    fatalError()
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func reversed(inAxes axes: [Int]) -> Tensor {
    fatalError()
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func reversed(inAxes axes: Int...) -> Tensor {
    reversed(inAxes: axes)
  }

  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public func concatenated(with other: Tensor, alongAxis axis: Int = 0) -> Tensor {
    fatalError()
  }

  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public static func ++ (lhs: Tensor, rhs: Tensor) -> Tensor {
    return lhs
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func gathering<Index: TensorFlowIndex>(
    atIndices indices: Tensor<Index>,
    alongAxis axis: Int = 0
  ) -> Tensor {
    fatalError()
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func batchGathering<Index: TensorFlowIndex>(
    atIndices indices: Tensor<Index>,
    alongAxis axis: Int = 1,
    batchDimensionCount: Int = 1
  ) -> Tensor {
    return self
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func gathering(where mask: Tensor<Bool>, alongAxis axis: Int = 0) -> Tensor {
    return self
  }
}

@usableFromInline
@noDerivative
internal func invertPermutationArray<T: TensorFlowIndex>(_ permutation: [T]) -> [T] {
  fatalError()
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: transposed(permutation:))
  func _vjpTransposed(permutation: Tensor<Int32>) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    fatalError()
  }

  @inlinable
  @derivative(of: transposed(permutation:))
  func _vjpTransposed(permutation: [Int]) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    fatalError()
  }

  @inlinable
  @derivative(of: transposed(permutation:))
  func _vjpTransposed(permutation: Int...) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    fatalError()
  }

  @inlinable
  @derivative(of: transposed)
  func _vjpTransposed() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    fatalError()
  }

  @inlinable
  @derivative(of: reversed)
  func _vjpReversed(inAxes axes: Tensor<Int32>) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    fatalError()
  }

  @inlinable
  @derivative(of: reversed)
  func _vjpReversed(inAxes axes: [Int]) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    fatalError()
  }

  @inlinable
  @derivative(of: reversed)
  func _vjpReversed(inAxes axes: Int...) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    fatalError()
  }

  @inlinable
  @derivative(of: concatenated)
  func _vjpConcatenated(
    with other: Tensor,
    alongAxis axis: Int
  ) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
    fatalError()
  }

  @inlinable
  @derivative(of: gathering)
  func _vjpGathering<Index: TensorFlowIndex>(
    atIndices indices: Tensor<Index>,
    alongAxis axis: Int = 0
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    fatalError()
  }
}

extension Tensor {
  @inlinable
  public func nonZeroIndices() -> Tensor<Int64> {
    fatalError()
  }
}


infix operator .=

extension Tensor {
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func broadcasted(toShape shape: Tensor<Int32>) -> Tensor {
    fatalError()
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func broadcasted(to shape: TensorShape) -> Tensor {
  return self
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func broadcasted<OtherScalar>(like other: Tensor<OtherScalar>) -> Tensor {
  return self
  }

  @inlinable
  public static func .= (lhs: inout Tensor, rhs: Tensor) {
    fatalError()
  }
}

extension Tensor where Scalar: Numeric {
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func unbroadcasted(toShape otherShape: Tensor<Int32>) -> Tensor {
  return self
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func unbroadcasted<OtherScalar>(like other: Tensor<OtherScalar>) -> Tensor {
  return self
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func unbroadcasted(to shape: TensorShape) -> Tensor {
    fatalError()
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: broadcasted)
  func _vjpBroadcasted(toShape shape: Tensor<Int32>) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    fatalError()
  }

  @inlinable
  @derivative(of: unbroadcasted)
  func _vjpUnbroadcasted(to shape: TensorShape) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    fatalError()
  }
}


extension Tensor where Scalar: Numeric {
  public enum PaddingMode {
    case constant(Scalar)
    case reflect
    case symmetric
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func padded(forSizes sizes: [(before: Int, after: Int)], with value: Scalar = 0)
    -> Tensor
  {
  return self
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func padded(forSizes sizes: [(before: Int, after: Int)], mode: PaddingMode) -> Tensor {
    fatalError()
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: padded)
  func _vjpPadded(
    forSizes sizes: [(before: Int, after: Int)],
    mode: PaddingMode
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    fatalError()
  }
}



extension Tensor {
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func slice(lowerBounds: [Int], upperBounds: [Int]) -> Tensor {
    return self
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func slice(lowerBounds: Tensor<Int32>, sizes: Tensor<Int32>) -> Tensor {
    fatalError()
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public func slice(lowerBounds: [Int], sizes: [Int]) -> Tensor {
    fatalError()
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: slice)
  internal func _vjpSlice(
    lowerBounds: Tensor<Int32>,
    sizes: Tensor<Int32>
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    fatalError()
  }

  @inlinable
  @derivative(of: slice(lowerBounds:sizes:))
  internal func _vjpSlice(
    lowerBounds: [Int],
    sizes: [Int]
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    fatalError()
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
    fatalError()
  }
}

public protocol TensorRangeExpression {
  var tensorRange: TensorRange { get }
}


extension Int: TensorRangeExpression {
  public var tensorRange: TensorRange {fatalError()}
}

extension Range: TensorRangeExpression where Bound == Int {
  public var tensorRange: TensorRange {
    fatalError()
  }
}

extension ClosedRange: TensorRangeExpression where Bound == Int {
  public var tensorRange: TensorRange {
    fatalError()
  }
}

extension PartialRangeFrom: TensorRangeExpression where Bound == Int {
  public var tensorRange: TensorRange {
    fatalError()
  }
}

extension PartialRangeUpTo: TensorRangeExpression where Bound == Int {
  public var tensorRange: TensorRange {
    fatalError()
  }
}

extension PartialRangeThrough: TensorRangeExpression where Bound == Int {
  public var tensorRange: TensorRange {
    fatalError()
  }
}

infix operator ..: StridedRangeFormationPrecedence
precedencegroup StridedRangeFormationPrecedence {
  associativity: left
  higherThan: CastingPrecedence
  lowerThan: RangeFormationPrecedence
}

extension Range where Bound == Int {
  public static func .. (range: Range, stride: Int) -> TensorRange {
    return .range(range, stride: stride)
  }
}

extension ClosedRange where Bound == Int {
  public static func .. (range: ClosedRange, stride: Int) -> TensorRange {
    return .closedRange(range, stride: stride)
  }
}

extension PartialRangeFrom where Bound == Int {
  public static func .. (range: PartialRangeFrom, stride: Int) -> TensorRange {
    return .partialRangeFrom(range, stride: stride)
  }
}

extension PartialRangeUpTo where Bound == Int {
  public static func .. (range: PartialRangeUpTo, stride: Int) -> TensorRange {
    fatalError()
  }
}

extension PartialRangeThrough where Bound == Int {
  public static func .. (range: PartialRangeThrough, stride: Int) -> TensorRange {
    fatalError()
  }
}

extension Tensor {
  @frozen @usableFromInline
  internal struct IndexPath {
    @usableFromInline
    let begin, end, strides: [Int32]

    @usableFromInline
    let beginMask, endMask, ellipsisMask, newAxisMask, squeezeAxisMask: Int64

    @inlinable
    public init(
      begin: [Int32], end: [Int32], strides: [Int32],
      beginMask: Int64, endMask: Int64, ellipsisMask: Int64, newAxisMask: Int64,
      squeezeAxisMask: Int64
    ) {
      fatalError()
    }
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  internal subscript(_ indexPath: IndexPath) -> Tensor {
    get {
      fatalError()
    }
    set {
      fatalError()
    }
  }

  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  public subscript(_ ranges: TensorRangeExpression...) -> Tensor {
    get { self }
    set {
      fatalError()
    }
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @usableFromInline
  @derivative(of: subscript)
  internal func _vjpSubscript(
    _ indexPath: IndexPath
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    fatalError()
  }
}

extension Tensor.IndexPath {
  @inlinable
  init(_ ranges: [TensorRange]) {
    fatalError()
  }
}


extension Tensor {
  @usableFromInline
  internal func isValid<T: BinaryInteger>(axis k: T) -> Bool {
    fatalError()
  }

  @usableFromInline
  internal func areValid<T: BinaryInteger>(axes: [T]) -> Bool {
    fatalError()
  }

  @usableFromInline
  internal func areValid(
    axes: Tensor<Int32>,
    file: StaticString = #file,
    line: UInt = #line
  ) -> Bool {
    fatalError()
  }

  @usableFromInline
  func ensureValid(
    axes: Tensor<Int32>,
    function: StaticString = #function,
    file: StaticString = #file,
    line: UInt = #line
  ) {
    fatalError()
  }

  @usableFromInline
  func ensureValid(
    axes: [Int],
    function: StaticString = #function,
    file: StaticString = #file,
    line: UInt = #line
  ) {
    fatalError()
  }

  @usableFromInline
  func ensureValid(
    axis k: Int,
    function: StaticString = #function,
    file: StaticString = #file,
    line: UInt = #line
  ) {
    fatalError()
  }
}
