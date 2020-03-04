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

import Swift

//===------------------------------------------------------------------------------------------===//
// TensorBuffer
//===------------------------------------------------------------------------------------------===//

/// `TensorBuffer` is the internal storage of `ShapedArray`. This buffer has two modes of storage:
/// `native` and `tensorFlow`. In `native` mode, the buffer object stores a pointer to contiguous
/// scalars; in `tensorFlow` mode, the buffer object stores a `TF_Tensor*` and bridges to
/// TensorFlow. In either mode, the buffer object owns the memory and will deallocate it on
/// `deinit`.
@usableFromInline
internal class TensorBuffer<Scalar> {
  /// Cached element count of the underlying buffer.
  let count: Int

  init(count: Int) { self.count = count }

  func withUnsafeBufferPointer<R>(
    _ body: (UnsafeBufferPointer<Scalar>) throws -> R
  ) rethrows -> R {
    fatalError("withUnsafeBufferPointer unimplemented because TensorBuffer is abstract")
  }

  func withUnsafeMutableBufferPointer<R>(
    _ body: (inout UnsafeMutableBufferPointer<Scalar>) throws -> R
  ) rethrows -> R {
    fatalError("withUnsafeMutableBufferPointer unimplemented because TensorBuffer is abstract")
  }
}

// TensorBuffer backed by a native swift array.
internal class ArrayTensorBuffer<Scalar>: TensorBuffer<Scalar> {
  var array: [Scalar]

  init(_ array: __owned [Scalar]) {
    self.array = array
    super.init(count: array.count)
  }

  override func withUnsafeBufferPointer<R>(
    _ body: (UnsafeBufferPointer<Scalar>) throws -> R
  ) rethrows -> R {
    return try array.withUnsafeBufferPointer(body)
  }

  override func withUnsafeMutableBufferPointer<R>(
    _ body: (inout UnsafeMutableBufferPointer<Scalar>) throws -> R
  ) rethrows -> R {
    return try array.withUnsafeMutableBufferPointer(body)
  }
}

//===------------------------------------------------------------------------------------------===//
// ShapedArrayProtocol: The protocol unifying ShapedArray and ShapedArraySlice.
//===------------------------------------------------------------------------------------------===//

public protocol _ShapedArrayProtocol: RandomAccessCollection, MutableCollection {
  associatedtype Scalar

  /// The number of dimensions of the array.
  var rank: Int { get }
  /// The shape of the array.
  var shape: [Int] { get }
  /// The total number of scalars in the array.
  var scalarCount: Int { get }

  /// Creates an array with the specified shape and contiguous scalars in row-major order.
  /// - Precondition: The number of scalars must equal the product of the dimensions of the shape.
  init(shape: [Int], scalars: [Scalar])

  /// Creates an array with the specified shape and sequence of scalars in row-major order.
  /// - Precondition: The number of scalars must equal the product of the dimensions of the shape.
  init<S: Sequence>(shape: [Int], scalars: S) where S.Element == Scalar

  /// Calls a closure with a pointer to the array’s contiguous storage.
  /// - Parameter body: A closure with an `UnsafeBufferPointer` parameter that points to the
  ///   contiguous storage for the array. If no such storage exists, it is created. If body has a
  ///   return value, that value is also used as the return value for the
  ///   `withUnsafeBufferPointer(_:)` method. The pointer argument is valid only for the duration
  ///   of the method's execution.
  func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Scalar>) throws -> R) rethrows -> R

  /// Calls the given closure with a pointer to the array’s mutable contiguous storage.
  /// - Parameter body: A closure with an `UnsafeMutableBufferPointer` parameter that points to
  ///   the contiguous storage for the array. If no such storage exists, it is created. If body
  ///   has a return value, that value is also used as the return value for the
  ///   `withUnsafeMutableBufferPointer(_:)` method. The pointer argument is valid only for the
  ///   duration of the method's execution.
  mutating func withUnsafeMutableBufferPointer<R>(
    _ body: (inout UnsafeMutableBufferPointer<Scalar>) throws -> R
  ) rethrows -> R
}

extension _ShapedArrayProtocol {
  /// The scalars of the array in row-major order.
  public var scalars: [Scalar] {
    get {
      return withUnsafeBufferPointer(Array.init)
    }
    set {
      precondition(newValue.count == scalarCount, "Scalar count mismatch.")
      withUnsafeMutableBufferPointer { pointer in
        pointer.baseAddress!.initialize(from: newValue, count: newValue.count)
      }
    }
  }

  /// Returns `true` if the array has rank 0.
  public var isScalar: Bool {
    return rank == 0
  }

  /// Returns the single scalar element if the array has rank 0 and `nil` otherwise.
  public var scalar: Scalar? {
    get {
      guard rank == 0 else { return nil }
      return scalars.first
    }
    set {
      precondition(isScalar, "Array does not have shape [].")
      guard let newValue = newValue else {
        preconditionFailure("New scalar value cannot be nil.")
      }
      scalars[0] = newValue
    }
  }
}

extension _ShapedArrayProtocol where Scalar: Equatable {
  public static func == <Other>(lhs: Self, rhs: Other) -> Bool
  where Other: _ShapedArrayProtocol, Scalar == Other.Scalar {
    return lhs.shape == rhs.shape && lhs.scalars.elementsEqual(rhs.scalars)
  }
}

extension _ShapedArrayProtocol {
  /// Returns the number of element arrays in an array (equivalent to the first dimension).
  /// - Note: `count` is distinct from `scalarCount`, which represents the
  ///   total number of scalars.
  public var count: Int {
    return shape.first ?? 0
  }
}

extension _ShapedArrayProtocol {
  /// Returns the scalar count for an element of the array.
  var scalarCountPerElement: Int {
    return shape.isEmpty ? 0 : shape.dropFirst().reduce(1, *)
  }

  /// Returns the scalar index corresponding to an index in the leading dimension of the array.
  func scalarIndex(fromIndex index: Int) -> Int {
    return scalarCountPerElement * index
  }

  /// Returns the range of scalars corresponding to a range in the leading dimension of the array.
  func scalarSubrange(from arraySubrange: Range<Int>) -> Range<Int> {
    return scalarIndex(
      fromIndex: arraySubrange.lowerBound)..<scalarIndex(fromIndex: arraySubrange.upperBound)
  }
}

extension String {
  /// Returns a string of the specified length, padded with whitespace to the left.
  fileprivate func leftPadded(toLength length: Int) -> String {
    return repeatElement(" ", count: max(0, length - count)) + self
  }
}

// Common public protocol implementations.

extension _ShapedArrayProtocol
where Element: _ShapedArrayProtocol, Element == Element.Element {
  /// Returns the whitespace separator between elements, given the current indent level.
  fileprivate func separator(indentLevel: Int) -> String {
    if rank == 1 {
      return ", "
    }
    return String(repeating: "\n", count: rank - 1)
      + String(repeating: " ", count: indentLevel + 1)
  }

  /// A textual representation of the 1-D shaped array, starting at the given indent level.
  /// Returns a summarized description if `summarizing` is true and the element count exceeds
  /// twice the `edgeElementCount`.
  ///
  /// - Parameters:
  ///   - indentLevel: The indentation level.
  ///   - edgeElementCount: The maximum number of elements to print before and after summarization
  ///     via ellipses (`...`).
  ///   - maxScalarLength: The length of the longest scalar description in the entire original
  ///     array-to-print.
  ///   - maxScalarCountPerLine: The maximum number of scalars to print per line, used when
  ///     printing 1-D vectors.
  ///   - summarizing: If true, summarize description if element count exceeds twice
  ///     `edgeElementCount`.
  fileprivate func vectorDescription(
    indentLevel: Int,
    edgeElementCount: Int,
    maxScalarLength: Int,
    maxScalarCountPerLine: Int,
    summarizing: Bool
  ) -> String {
    // Get scalar descriptions.
    func scalarDescription(_ element: Element) -> String {
      let description = String(describing: element)
      return description.leftPadded(toLength: maxScalarLength)
    }

    var scalarDescriptions: [String] = []
    if summarizing && count > 2 * edgeElementCount {
      scalarDescriptions += prefix(edgeElementCount).map(scalarDescription)
      scalarDescriptions += ["..."]
      scalarDescriptions += suffix(edgeElementCount).map(scalarDescription)
    } else {
      scalarDescriptions += map(scalarDescription)
    }

    // Combine scalar descriptions into lines, based on the scalar count per line.
    let lines = stride(
      from: scalarDescriptions.startIndex,
      to: scalarDescriptions.endIndex,
      by: maxScalarCountPerLine
    ).map { i -> ArraySlice<String> in
      let upperBound = Swift.min(
        i.advanced(by: maxScalarCountPerLine),
        scalarDescriptions.count)
      return scalarDescriptions[i..<upperBound]
    }

    // Return lines joined with separators.
    let lineSeparator = ",\n" + String(repeating: " ", count: indentLevel + 1)
    return lines.enumerated().reduce(into: "[") { result, entry in
      let (i, line) = entry
      result += line.joined(separator: ", ")
      result += i != lines.count - 1 ? lineSeparator : ""
    } + "]"
  }

  /// A textual representation of the shaped array, starting at the given indent level. Returns a
  /// summarized description if `summarizing` is true and the element count exceeds twice the
  /// `edgeElementCount`.
  ///
  /// - Parameters:
  ///   - indentLevel: The indentation level.
  ///   - edgeElementCount: The maximum number of elements to print before and after summarization
  ///     via ellipses (`...`).
  ///   - maxScalarLength: The length of the longest scalar description in the entire original
  ///     array-to-print.
  ///   - maxScalarCountPerLine: The maximum number of scalars to print per line, used when
  ///     printing 1-D vectors.
  ///   - summarizing: If true, summarizing description if element count exceeds twice
  ///     `edgeElementCount`.
  fileprivate func description(
    indentLevel: Int,
    edgeElementCount: Int,
    maxScalarLength: Int,
    maxScalarCountPerLine: Int,
    summarizing: Bool
  ) -> String {
    // Handle scalars.
    if let scalar = scalar {
      return String(describing: scalar)
    }

    // Handle vectors, which have special line-width-sensitive logic.
    if rank == 1 {
      return vectorDescription(
        indentLevel: indentLevel,
        edgeElementCount: edgeElementCount,
        maxScalarLength: maxScalarLength,
        maxScalarCountPerLine: maxScalarCountPerLine,
        summarizing: summarizing)
    }

    // Handle higher-rank tensors.
    func elementDescription(_ element: Element) -> String {
      return element.description(
        indentLevel: indentLevel + 1,
        edgeElementCount: edgeElementCount,
        maxScalarLength: maxScalarLength,
        maxScalarCountPerLine: maxScalarCountPerLine,
        summarizing: summarizing)
    }

    var elementDescriptions: [String] = []
    if summarizing && count > 2 * edgeElementCount {
      elementDescriptions += prefix(edgeElementCount).map(elementDescription)
      elementDescriptions += ["..."]
      elementDescriptions += suffix(edgeElementCount).map(elementDescription)
    } else {
      elementDescriptions += map(elementDescription)
    }

    // Return lines joined with separators.
    let lineSeparator =
      "," + String(repeating: "\n", count: rank - 1)
      + String(repeating: " ", count: indentLevel + 1)
    return elementDescriptions.enumerated().reduce(into: "[") { result, entry in
      let (i, elementDescription) = entry
      result += elementDescription
      result += i != elementDescriptions.count - 1 ? lineSeparator : ""
    } + "]"
  }
}

extension _ShapedArrayProtocol
where Element: _ShapedArrayProtocol, Element == Element.Element {
  /// A textual representation of the shaped array. Returns a summarized description if
  /// `summarizing` is true and the element count exceeds twice the `edgeElementCount`.
  ///
  /// - Parameters:
  ///   - lineWidth: The max line width for printing. Used to determine number of scalars to print
  ///     per line.
  ///   - edgeElementCount: The maximum number of elements to print before and after summarization
  ///     via ellipses (`...`).
  ///   - summarizing: If true, summarizing description if element count exceeds twice
  ///     `edgeElementCount`.
  public func description(
    lineWidth: Int = 80,
    edgeElementCount: Int = 3,
    summarizing: Bool = false
  ) -> String {
    // Compute the number of scalars to print per line.
    let maxScalarLength = scalars.lazy.map { String(describing: $0).count }.max() ?? 3
    let maxScalarCountPerLine = Swift.max(1, lineWidth / maxScalarLength)
    return description(
      indentLevel: 0,
      edgeElementCount: edgeElementCount,
      maxScalarLength: maxScalarLength,
      maxScalarCountPerLine: maxScalarCountPerLine,
      summarizing: summarizing)
  }

  /// A full, non-pretty-printed textual representation of the shaped array, showing all scalars.
  public var fullDescription: String {
    if let scalar = scalar {
      return String(describing: scalar)
    }
    return "[\( map({"\($0.fullDescription)"}).joined(separator: ", ") )]"
  }
}

//===------------------------------------------------------------------------------------------===//
// ShapedArray
//===------------------------------------------------------------------------------------------===//

/// `ShapedArray` is a multi-dimensional array. It has a shape, which has type `[Int]` and defines
/// the array dimensions, and uses a `TensorBuffer` internally as storage.
@frozen
public struct ShapedArray<Scalar>: _ShapedArrayProtocol {
  /// Contiguous memory storing scalars.
  internal var buffer: TensorBuffer<Scalar>

  /// The dimensions of the array.
  public private(set) var shape: [Int]

  /// Creates a `ShapedArray` from a `TensorBuffer` and a shape.
  internal init(buffer: __owned TensorBuffer<Scalar>, shape: __owned [Int]) {
    precondition(
      buffer.count == shape.reduce(1, *),
      "The scalar count of the buffer does not match the shape.")
    self.buffer = buffer
    self.shape = shape
    debugLog("Done initializing ShapedArray from TensorBuffer.")
  }
}

extension ShapedArray {
  fileprivate mutating func ensureUniquelyReferenced() {
    if isKnownUniquelyReferenced(&buffer) { return }
    let oldBuffer = buffer
    debugLog("Unique reference check")
    buffer = oldBuffer.withUnsafeBufferPointer { oldBufferPointer in
      ArrayTensorBuffer<Scalar>([Scalar](oldBufferPointer))
    }
  }
}

extension ShapedArray {
  /// The number of dimensions of the array.
  public var rank: Int {
    return shape.count
  }

  /// The total number of scalars in the array.
  public var scalarCount: Int {
    return buffer.count
  }

  /// Creates a `ShapedArray` with the same shape and scalars as the specified instance.
  public init(_ other: ShapedArray) {
    debugLog("Initializing from another ShapedArray.")
    self.init(buffer: other.buffer, shape: other.shape)
  }

  /// Creates a `ShapedArray` with the specified shape and contiguous scalars in row-major order.
  /// - Precondition: The number of scalars must equal the product of the dimensions of the shape.
  public init(shape: __owned [Int], scalars: __owned [Scalar]) {
    precondition(shape.reduce(1, *) == scalars.count, "Scalar count mismatch.")
    let buffer = ArrayTensorBuffer<Scalar>(scalars)
    self.init(buffer: buffer, shape: shape)
  }

  /// Creates a `ShapedArray` with the specified shape and sequence of scalars in row-major order.
  /// - Precondition: The number of scalars must equal the product of the dimensions of the shape.
  public init<S: Sequence>(shape: __owned [Int], scalars: __shared S) where S.Element == Scalar {
    let scalarCount = shape.reduce(1, *)
    let buffer = ArrayTensorBuffer<Scalar>([Scalar](scalars))
    precondition(
      buffer.count == scalarCount,
      "The sequence has fewer elements than needed by the shape.")
    self.init(buffer: buffer, shape: shape)
  }

  /// Creates a `ShapedArray` from a scalar value.
  public init(_ scalar: __owned Scalar) {
    self.init(buffer: ArrayTensorBuffer([scalar]), shape: [])
  }

  /// Creates a `ShapedArray` with the specified shape and a single, repeated scalar value.
  /// - Parameters:
  ///   - shape: The shape of the `ShapedArray`.
  ///   - repeatedValue: The scalar value to repeat.
  @inlinable
  @available(*, deprecated, renamed: "init(repeating:shape:)")
  public init(shape: __owned [Int], repeating repeatedValue: __owned Scalar) {
    self.init(repeating: repeatedValue, shape: shape)
  }

  /// Creates a `ShapedArray` with the specified shape and a single, repeated scalar value.
  /// - Parameters:
  ///   - repeatedValue: The scalar value to repeat.
  ///   - shape: The shape of the `ShapedArray`.
  public init(repeating repeatedValue: __owned Scalar, shape: __owned [Int]) {
    let scalarCount = shape.reduce(1, *)
    let buffer = ArrayTensorBuffer<Scalar>(Array(repeating: repeatedValue, count: scalarCount))
    self.init(buffer: buffer, shape: shape)
  }
}

extension ShapedArray: RandomAccessCollection, MutableCollection {
  public typealias Index = Int
  public typealias Element = ShapedArraySlice<Scalar>
  public typealias SubSequence = ShapedArraySlice<Scalar>

  public var indices: Range<Int> {
    return 0..<count
  }

  public var startIndex: Int {
    return 0
  }

  public var endIndex: Int {
    return count
  }

  /// Access the element array specified by an index in the leading dimension.
  /// - Parameter index: Index of the element array.
  public subscript(index: Int) -> Element {
    get {
      precondition(!isScalar, "Scalar has no elements and cannot be subscripted.")
      precondition(index < endIndex, "ShapedArray index is out of range.")
      precondition(index >= startIndex, "Negative ShapedArray index is out of range.")
      return ShapedArraySlice(base: self, baseIndices: [index])
    }
    set {
      precondition(!isScalar, "Scalar has no elements and cannot be subscripted.")
      precondition(index < endIndex, "ShapedArray index is out of range.")
      precondition(index >= startIndex, "Negative ShapedArray index is out of range.")
      precondition(shape.dropFirst().elementsEqual(newValue.shape), "Element shape mismatch.")
      let scalarIndex = self.scalarIndex(fromIndex: index)
      withUnsafeMutableBufferPointer { destBuffPtr in
        let ptr = destBuffPtr.baseAddress!.advanced(by: scalarIndex)
        newValue.withUnsafeBufferPointer { srcBuffPtr in
          ptr.initialize(from: srcBuffPtr.baseAddress!, count: srcBuffPtr.count)
        }
      }
    }
  }

  /// Access the subarray specified by a contiguous range of indices.
  /// - Parameter bounds: Contiguous range of indices.
  public subscript(bounds: Range<Int>) -> SubSequence {
    get {
      precondition(!isScalar, "Scalar has no elements and cannot be subscripted.")
      precondition(
        bounds.lowerBound >= startIndex && bounds.lowerBound <= endIndex
          && bounds.upperBound >= startIndex && bounds.upperBound <= endIndex,
        "ShapedArray indices are out of range")
      return ShapedArraySlice(base: self, bounds: bounds)
    }
    set {
      precondition(!isScalar, "Scalar has no elements and cannot be subscripted.")
      precondition(
        indices ~= bounds.lowerBound && indices ~= bounds.upperBound - 1,
        "ShapedArray indices are out of range.")
      let subArrayShape = [bounds.count] + shape.dropFirst()
      precondition(subArrayShape == newValue.shape, "Subarray shape mismatch.")
      let scalarIndex = self.scalarIndex(fromIndex: bounds.lowerBound)
      withUnsafeMutableBufferPointer { destBuffPtr in
        let ptr = destBuffPtr.baseAddress!.advanced(by: scalarIndex)
        newValue.withUnsafeBufferPointer { srcBuffPtr in
          ptr.initialize(from: srcBuffPtr.baseAddress!, count: srcBuffPtr.count)
        }
      }
    }
  }
}

extension ShapedArray {
  /// Calls a closure with a pointer to the array’s contiguous storage.
  /// - Parameter body: A closure with an `UnsafeBufferPointer` parameter that points to the
  ///   contiguous storage for the array. If no such storage exists, it is created. If body has a
  ///   return value, that value is also used as the return value for the
  ///   `withUnsafeBufferPointer(_:)` method. The pointer argument is valid only for the duration
  ///   of the method's execution.
  public func withUnsafeBufferPointer<Result>(
    _ body: (UnsafeBufferPointer<Scalar>) throws -> Result
  ) rethrows -> Result {
    return try buffer.withUnsafeBufferPointer { ptr in try body(ptr) }
  }

  /// Calls the given closure with a pointer to the array’s mutable contiguous storage.
  /// - Parameter body: A closure with an `UnsafeMutableBufferPointer` parameter that points to
  ///   the contiguous storage for the array. If no such storage exists, it is created. If body
  ///   has a return value, that value is also used as the return value for the
  ///   `withUnsafeMutableBufferPointer(_:)` method. The pointer argument is valid only for the
  ///   duration of the method's execution.
  public mutating func withUnsafeMutableBufferPointer<Result>(
    _ body: (inout UnsafeMutableBufferPointer<Scalar>) throws -> Result
  ) rethrows -> Result {
    ensureUniquelyReferenced()
    return try buffer.withUnsafeMutableBufferPointer { ptr in try body(&ptr) }
  }
}

// Array literal conversion.
extension ShapedArray: ExpressibleByArrayLiteral where Scalar: TensorFlowScalar {
  public typealias ArrayLiteralElement = _TensorElementLiteral<Scalar>
  @inlinable
  public init(arrayLiteral elements: _TensorElementLiteral<Scalar>...) {
    precondition(!elements.isEmpty, "Cannot create a 'ShapedArray' with no elements.")
    self = Tensor<Scalar>(_tensorElementLiterals: elements).array
  }
}

// Equatable conformance.
extension ShapedArray: Equatable where Scalar: Equatable {
  public static func == (lhs: ShapedArray, rhs: ShapedArray) -> Bool {
    return lhs._isEqual(to: rhs)
  }
}

// Hashable conformance.
extension ShapedArray: Hashable where Scalar: Hashable {
  public func hash(into hasher: inout Hasher) {
    hasher.combine(shape)
    hasher.combine(scalars)
  }
}

// String conversion.
extension ShapedArray: CustomStringConvertible {
  /// A textual representation of this `ShapedArray`.
  ///
  /// - Note: use `fullDescription` for a non-pretty-printed description showing all scalars.
  public var description: String {
    // Summarize if there are more than 1000 scalars.
    let summarizing = scalarCount > 1000
    return description(summarizing: summarizing)
  }
}

// Xcode Playground display conversion.
extension ShapedArray: CustomPlaygroundDisplayConvertible {
  public var playgroundDescription: Any {
    return description
  }
}

// Mirror representation, used by debugger/REPL.
extension ShapedArray: CustomReflectable {
  public var customMirror: Mirror {
    return Mirror(self, children: [], displayStyle: .struct)
  }
}

// Codable conformance.
extension ShapedArray: Codable where Scalar: Codable {
  private enum CodingKeys: String, CodingKey {
    case shape
    case scalars
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let shape = try container.decode([Int].self, forKey: .shape)
    let scalars = try container.decode([Scalar].self, forKey: .scalars)
    self.init(shape: shape, scalars: scalars)
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(shape, forKey: .shape)
    try container.encode(scalars, forKey: .scalars)
  }
}

//===------------------------------------------------------------------------------------------===//
// ShapedArraySlice
//===------------------------------------------------------------------------------------------===//

/// A contiguous slice of a `ShapedArray` or `ShapedArraySlice` instance.
///
/// `ShapedArraySlice` enables fast, efficient operations on contiguous slices of `ShapedArray`
/// instances. `ShapedArraySlice` instances do not have their own storage. Instead, they provides a
/// view onto the storage of their base `ShapedArray`. `ShapedArraySlice` can represent two
/// different kinds of slices: element arrays and subarrays.
///
/// Element arrays are subdimensional elements of a `ShapedArray`: their rank is one less than that
/// of their base. Element array slices are obtained by indexing a `ShapedArray` instance with a
/// singular `Int32` index.
///
/// For example:
/// ```
///     var matrix = ShapedArray(shape: [2, 2], scalars: [0, 1, 2, 3])
///     // `matrix` represents [[0, 1], [2, 3]].
///
///     let element = matrix[0]
///     // `element` is a `ShapedArraySlice` with shape [2]. It is an element
///     // array, specifically the first element in `matrix`: [0, 1].
///
///     matrix[1] = ShapedArraySlice(shape: [2], scalars: [4, 8])
///     // The second element in `matrix` has been mutated.
///     // `matrix` now represents [[0, 1, 4, 8]].
/// ```
///
/// Subarrays are a contiguous range of the elements in a `ShapedArray`. The rank of a subarray is
/// the same as that of its base, but its leading dimension is the count of the slice range.
/// Subarray slices are obtained by indexing a `ShapedArray` with a `Range<Int32>` that represents a
/// range of elements (in the leading dimension). Methods like `prefix(:)` and `suffix(:)` that
/// internally index with a range also produce subarray.
///
/// For example:
/// ```
///     let zeros = ShapedArray(repeating: 0, shape: [3, 2])
///     var matrix = ShapedArray(shape: [3, 2], scalars: Array(0..<6))
///     // `zeros` represents [[0, 0], [0, 0], [0, 0]].
///     // `matrix` represents [[0, 1], [2, 3], [4, 5]].
///
///     let subarray = matrix.prefix(2)
///     // `subarray` is a `ShapedArraySlice` with shape [2, 2]. It is a slice
///     // of the first 2 elements in `matrix` and represents [[0, 1], [2, 3]].
///
///     matrix[0..<2] = zeros.prefix(2)
///     // The first 2 elements in `matrix` have been mutated.
///     // `matrix` now represents [[0, 0], [0, 0], [4, 5]].
/// ```
@frozen
public struct ShapedArraySlice<Scalar>: _ShapedArrayProtocol {
  /// The underlying `ShapedArray` of the slice.
  @usableFromInline internal var base: ShapedArray<Scalar>
  /// The subdimensional indices of a slice.
  @usableFromInline internal var baseIndices: [Int]
  /// The subarray bounds of a slice.
  @usableFromInline internal var bounds: Range<Int>?

  /// Creates a `ShapedArraySlice` from a base `ShapedArray`, with the specified subdimensional
  /// indices and subarray bounds.
  @inlinable
  internal init(
    base: __owned ShapedArray<Scalar>,
    baseIndices indices: __owned [Int] = [],
    bounds: Range<Int>? = nil
  ) {
    precondition(indices.count <= base.rank, "Number of base indices exceeds base rank.")
    precondition(
      zip(base.shape, indices).allSatisfy { $1 >= 0 && $1 < $0 },
      "Base indices are out of range")
    self.base = base
    self.baseIndices = indices
    self.bounds = bounds
  }
}

extension ShapedArraySlice {
  /// Indexing depth of this slice, i.e. the difference in rank between the base and the slice.
  internal var indexingDepth: Int {
    return baseIndices.count
  }

  /// The number of dimensions of the array.
  public var rank: Int {
    return base.rank - indexingDepth
  }

  /// The shape of the array.
  public var shape: [Int] {
    if let bounds = bounds {
      return [bounds.count] + Array(base.shape.dropFirst(indexingDepth + 1))
    }
    return Array(base.shape.dropFirst(indexingDepth))
  }

  /// The total number of scalars in the array.
  public var scalarCount: Int {
    return shape.reduce(1, *)
  }
}

// Slice initializers.
extension ShapedArraySlice {
  /// Creates a `ShapedArraySlice` with the specified shape and contiguous scalars in row-major
  /// order.
  /// - Precondition: The number of scalars must equal the product of the dimensions of the shape.
  public init(shape: __owned [Int], scalars: __owned [Scalar]) {
    self.init(base: ShapedArray(shape: shape, scalars: scalars))
  }

  /// Creates an `ShapedArraySlice` with the specified shape and sequence of scalars in row-major
  /// order.
  /// - Precondition: The number of scalars must equal the product of the dimensions of the shape.
  public init<S: Sequence>(shape: __owned [Int], scalars: __shared S) where S.Element == Scalar {
    self.init(base: ShapedArray(shape: shape, scalars: scalars))
  }

  /// Creates a `ShapedArraySlice` from a scalar value.
  public init(_ scalar: __owned Scalar) {
    self.init(base: ShapedArray(scalar))
  }

  /// Creates a `ShapedArraySlice` with the specified shape and a single, repeated scalar value.
  /// - Parameters:
  ///   - repeatedValue: The scalar value to repeat.
  ///   - shape: The shape of the `ShapedArraySlice`.
  @inlinable
  @available(*, deprecated, renamed: "init(repeating:shape:)")
  public init(shape: __owned [Int], repeating repeatedValue: __owned Scalar) {
    self.init(repeating: repeatedValue, shape: shape)
  }

  /// Creates a `ShapedArraySlice` with the specified shape and a single, repeated scalar value.
  /// - Parameters:
  ///   - repeatedValue: The scalar value to repeat.
  ///   - shape: The shape of the `ShapedArraySlice`.
  public init(repeating repeatedValue: __owned Scalar, shape: __owned [Int]) {
    self.init(base: ShapedArray(repeating: repeatedValue, shape: shape))
  }
}

extension ShapedArraySlice {
  /// The range of scalars from the base `ShapedArray` represented by a `ShapedArraySlice`.
  var scalarRange: Range<Int> {
    let trimmedShape = base.shape.dropFirst()
    var (start, end) = baseIndices.enumerated().reduce((0, base.scalarCount)) { (acc, next) in
      let stride = trimmedShape.dropFirst(next.offset).reduce(1, *)
      if next.offset == indexingDepth - 1 {
        let temp = acc.0 + next.element * stride
        return (temp, temp + stride)
      }
      return (acc.0 + next.element * stride, acc.1)
    }
    if let bounds = bounds {
      let stride = trimmedShape.dropFirst(indexingDepth).reduce(1, *)
      let oldStart = start
      start = start + bounds.startIndex * stride
      end = oldStart + bounds.endIndex * stride
    }
    return start..<end
  }
}

extension ShapedArraySlice {
  /// Calls a closure with a pointer to the `ShapedArraySlice`’s contiguous storage.
  /// - Parameter body: A closure with an `UnsafeBufferPointer` parameter that points to the
  ///   contiguous storage for the `ShapedArraySlice`. If no such storage exists, it is created.
  ///   If body has a return value, that value is also used as the return value for the
  ///   `withUnsafeBufferPointer(_:)` method. The pointer argument is valid only for the duration
  ///   of the method's execution.
  public func withUnsafeBufferPointer<Result>(
    _ body: (UnsafeBufferPointer<Scalar>) throws -> Result
  ) rethrows -> Result {
    return try base.withUnsafeBufferPointer { baseBuffPtr in
      let basePtr = baseBuffPtr.baseAddress!
      let ptr = UnsafeBufferPointer(
        start: basePtr.advanced(by: scalarRange.startIndex),
        count: scalarRange.count)
      return try body(ptr)
    }
  }

  /// Calls the given closure with a pointer to the `ShapedArraySlice`'s mutable contiguous
  /// storage.
  /// - Parameter body: A closure with an `UnsafeMutableBufferPointer` parameter that points to
  ///   the contiguous storage for the `ShapedArraySlice`. If no such storage exists, it is
  ///   created. If body has a return value, that value is also used as the return value for the
  ///   `withUnsafeMutableBufferPointer(_:)` method. The pointer argument is valid only for the
  ///   duration of the method’s execution.
  public mutating func withUnsafeMutableBufferPointer<Result>(
    _ body: (inout UnsafeMutableBufferPointer<Scalar>) throws -> Result
  ) rethrows -> Result {
    // NOTE: Copying `scalarRange` to a local variable here is necessary for
    // exclusive access.
    let scalarRange = self.scalarRange
    return try base.withUnsafeMutableBufferPointer { baseBuffPtr in
      let basePtr = baseBuffPtr.baseAddress!
      var ptr = UnsafeMutableBufferPointer(
        start: basePtr.advanced(by: scalarRange.startIndex),
        count: scalarRange.count)
      return try body(&ptr)
    }
  }
}

extension ShapedArraySlice: RandomAccessCollection, MutableCollection {
  public typealias Index = Int
  public typealias Element = ShapedArraySlice
  public typealias SubSequence = ShapedArraySlice

  public var indices: Range<Int> {
    if let bounds = bounds {
      return bounds
    } else if indexingDepth < base.rank {
      return 0..<base.shape[indexingDepth]
    }
    return 0..<0
  }

  public var startIndex: Int {
    return indices.startIndex
  }

  public var endIndex: Int {
    return indices.endIndex
  }

  /// Access the element array specified by an index in the leading dimension.
  /// - Parameter index: Index of the element array.
  public subscript(index: Int) -> Element {
    get {
      precondition(!isScalar, "Scalar has no elements and cannot be subscripted.")
      precondition(index < endIndex, "ShapedArraySlice index is out of range.")
      precondition(
        index >= startIndex,
        "ShapeArraySlice index is out of range (before startIndex).")
      return ShapedArraySlice(base: base, baseIndices: baseIndices + [index], bounds: nil)
    }
    set {
      precondition(!isScalar, "Scalar has no elements and cannot be subscripted.")
      precondition(index < endIndex, "ShapedArraySlice index is out of range")
      precondition(
        index >= startIndex,
        "ShapeArraySlice index is out of range (before startIndex).")
      precondition(shape.dropFirst().elementsEqual(newValue.shape), "Element shape mismatch.")
      let scalarIndex = self.scalarIndex(fromIndex: index)
      withUnsafeMutableBufferPointer { destBuffPtr in
        let ptr = destBuffPtr.baseAddress!.advanced(by: scalarIndex)
        newValue.withUnsafeBufferPointer { srcBuffPtr in
          ptr.initialize(from: srcBuffPtr.baseAddress!, count: srcBuffPtr.count)
        }
      }
    }
  }

  /// Access the subarray specified by a contiguous range of indices.
  /// - Parameter bounds: Contiguous range of indices.
  public subscript(bounds: Range<Int>) -> SubSequence {
    get {
      precondition(!isScalar, "Scalar has no elements and cannot be subscripted.")
      precondition(
        indices ~= bounds.lowerBound && indices ~= bounds.upperBound - 1,
        "ShapedArraySlice indices are out of range.")
      return ShapedArraySlice(base: base, baseIndices: baseIndices, bounds: bounds)
    }
    set {
      precondition(!isScalar, "Scalar has no elements and cannot be subscripted.")
      precondition(
        indices ~= bounds.lowerBound && indices ~= bounds.upperBound - 1,
        "ShapedArraySlice indices are out of range.")
      let subArrayShape = [bounds.count] + shape.dropFirst()
      precondition(subArrayShape == newValue.shape, "Subarray shape mismatch.")
      let scalarIndex = self.scalarIndex(fromIndex: bounds.lowerBound)
      withUnsafeMutableBufferPointer { destBuffPtr in
        let ptr = destBuffPtr.baseAddress!.advanced(by: scalarIndex)
        newValue.withUnsafeBufferPointer { srcBuffPtr in
          ptr.initialize(from: srcBuffPtr.baseAddress!, count: srcBuffPtr.count)
        }
      }
    }
  }
}

// Tensor conversion.
extension ShapedArraySlice where Scalar: TensorFlowScalar {
  public init(_ tensor: __shared Tensor<Scalar>) {
    self.init(base: tensor.array)
  }
}

// Array literal conversion.
extension ShapedArraySlice: ExpressibleByArrayLiteral where Scalar: TensorFlowScalar {
  public typealias ArrayLiteralElement = _TensorElementLiteral<Scalar>
  @inlinable
  public init(arrayLiteral elements: _TensorElementLiteral<Scalar>...) {
    precondition(!elements.isEmpty, "Cannot create a 'ShapedArraySlice' with no elements.")
    self.init(base: Tensor(_tensorElementLiterals: elements).array)
  }
}

// Equatable conformance.
extension ShapedArraySlice: Equatable where Scalar: Equatable {
  public static func == (lhs: ShapedArraySlice, rhs: ShapedArraySlice) -> Bool {
    return lhs._isEqual(to: rhs)
  }
}

// Hashable conformance.
extension ShapedArraySlice: Hashable where Scalar: Hashable {
  public func hash(into hasher: inout Hasher) {
    hasher.combine(shape)
    hasher.combine(scalars)
  }
}

// String conversion.
extension ShapedArraySlice: CustomStringConvertible {
  /// A textual representation of this `ShapedArraySlice`.
  ///
  /// - Note: use `fullDescription` for a non-pretty-printed representation showing all scalars.
  public var description: String {
    // Summarize if there are more than 1000 scalars.
    let summarizing = scalarCount > 1000
    return description(summarizing: summarizing)
  }
}

// Xcode Playground display conversion.
extension ShapedArraySlice: CustomPlaygroundDisplayConvertible {
  public var playgroundDescription: Any {
    return description
  }
}

// Mirror representation, used by debugger/REPL.
extension ShapedArraySlice: CustomReflectable {
  public var customMirror: Mirror {
    return Mirror(self, children: [], displayStyle: .struct)
  }
}

// Codable conformance.
extension ShapedArraySlice: Codable where Scalar: Codable {
  private enum CodingKeys: String, CodingKey {
    case shape
    case scalars
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(shape, forKey: .shape)
    try container.encode(scalars, forKey: .scalars)
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let shape = try container.decode([Int].self, forKey: .shape)
    let scalars = try container.decode([Scalar].self, forKey: .scalars)
    self.init(shape: shape, scalars: scalars)
  }
}

extension _ShapedArrayProtocol where Scalar: Equatable {
  fileprivate func _isEqual(to other: Self) -> Bool {
    return shape == other.shape
      && withUnsafeBufferPointer { selfBuf in
        other.withUnsafeBufferPointer { otherBuf in
          selfBuf.elementsEqual(otherBuf)
        }
      }
  }
}
