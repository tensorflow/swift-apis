//===--- Algorithms.swift -------------------------------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//
// RUN: %empty-directory(%t)
// RUN: %target-build-swift -g -Onone -DUSE_STDLIBUNITTEST %s -o %t/a.out
// RUN: %target-codesign %t/a.out
// RUN: %target-run %t/a.out
// REQUIRES: executable_test

#if USE_STDLIBUNITTEST
  import Swift
  import StdlibUnittest
#endif

//===--- Rotate -----------------------------------------------------------===//
//===----------------------------------------------------------------------===//

/// Provides customization points for `MutableCollection` algorithms.
///
/// If incorporated into the standard library, these requirements would just be
/// part of `MutableCollection`.  In the meantime, you can declare conformance
/// of a collection to `MutableCollectionAlgorithms` to get these customization
/// points to be used from other algorithms defined on
/// `MutableCollectionAlgorithms`.
public protocol MutableCollectionAlgorithms: MutableCollection
where SubSequence: MutableCollectionAlgorithms {
  /// Rotates the elements of the collection so that the element
  /// at `middle` ends up first.
  ///
  /// - Returns: The new index of the element that was first
  ///   pre-rotation.
  /// - Complexity: O(*n*)
  @discardableResult
  mutating func rotate(shiftingToStart middle: Index) -> Index
}

// Conformances of common collection types to MutableCollectionAlgorithms.
// If rotate was a requirement of MutableCollection, these would not be needed.
extension Array: MutableCollectionAlgorithms {}
extension ArraySlice: MutableCollectionAlgorithms {}
extension Slice: MutableCollectionAlgorithms
where Base: MutableCollection {}

extension MutableCollection {
  /// Swaps the elements of the two given subranges, up to the upper bound of
  /// the smaller subrange. The returned indices are the ends of the two ranges
  /// that were actually swapped.
  ///
  ///     Input:
  ///     [a b c d e f g h i j k l m n o p]
  ///      ^^^^^^^         ^^^^^^^^^^^^^
  ///      lhs             rhs
  ///
  ///     Output:
  ///     [i j k l e f g h a b c d m n o p]
  ///             ^               ^
  ///             p               q
  ///
  /// - Precondition: !lhs.isEmpty && !rhs.isEmpty
  /// - Postcondition: For returned indices `(p, q)`:
  ///   - distance(from: lhs.lowerBound, to: p) ==
  ///       distance(from: rhs.lowerBound, to: q)
  ///   - p == lhs.upperBound || q == rhs.upperBound
  @inline(__always)
  internal mutating func _swapNonemptySubrangePrefixes(
    _ lhs: Range<Index>, _ rhs: Range<Index>
  ) -> (Index, Index) {
    assert(!lhs.isEmpty)
    assert(!rhs.isEmpty)

    var p = lhs.lowerBound
    var q = rhs.lowerBound
    repeat {
      swapAt(p, q)
      formIndex(after: &p)
      formIndex(after: &q)
    } while p != lhs.upperBound && q != rhs.upperBound
    return (p, q)
  }

  /// Rotates the elements of the collection so that the element
  /// at `middle` ends up first.
  ///
  /// - Returns: The new index of the element that was first
  ///   pre-rotation.
  /// - Complexity: O(*n*)
  @discardableResult
  public mutating func rotate(shiftingToStart middle: Index) -> Index {
    var m = middle
    var s = startIndex
    let e = endIndex

    // Handle the trivial cases
    if s == m { return e }
    if m == e { return s }

    // We have two regions of possibly-unequal length that need to be
    // exchanged.  The return value of this method is going to be the
    // position following that of the element that is currently last
    // (element j).
    //
    //   [a b c d e f g|h i j]   or   [a b c|d e f g h i j]
    //   ^             ^     ^        ^     ^             ^
    //   s             m     e        s     m             e
    //
    var ret = e  // start with a known incorrect result.
    while true {
      // Exchange the leading elements of each region (up to the
      // length of the shorter region).
      //
      //   [a b c d e f g|h i j]   or   [a b c|d e f g h i j]
      //    ^^^^^         ^^^^^          ^^^^^ ^^^^^
      //   [h i j d e f g|a b c]   or   [d e f|a b c g h i j]
      //   ^     ^       ^     ^         ^    ^     ^       ^
      //   s    s1       m    m1/e       s   s1/m   m1      e
      //
      let (s1, m1) = _swapNonemptySubrangePrefixes(s..<m, m..<e)

      if m1 == e {
        // Left-hand case: we have moved element j into position.  if
        // we haven't already, we can capture the return value which
        // is in s1.
        //
        // Note: the STL breaks the loop into two just to avoid this
        // comparison once the return value is known.  I'm not sure
        // it's a worthwhile optimization, though.
        if ret == e { ret = s1 }

        // If both regions were the same size, we're done.
        if s1 == m { break }
      }

      // Now we have a smaller problem that is also a rotation, so we
      // can adjust our bounds and repeat.
      //
      //    h i j[d e f g|a b c]   or    d e f[a b c|g h i j]
      //         ^       ^     ^              ^     ^       ^
      //         s       m     e              s     m       e
      s = s1
      if s == m { m = m1 }
    }

    return ret
  }
}

extension MutableCollection where Self: BidirectionalCollection {

  /// Reverses the elements of the collection, moving from each end until
  /// `limit` is reached from either direction. The returned indices are the
  /// start and end of the range of unreversed elements.
  ///
  ///     Input:
  ///     [a b c d e f g h i j k l m n o p]
  ///             ^
  ///           limit
  ///     Output:
  ///     [p o n m e f g h i j k l d c b a]
  ///             ^               ^
  ///             f               l
  ///
  /// - Postcondition: For returned indices `(f, l)`:
  ///   `f == limit || l == limit`
  @inline(__always)
  @discardableResult
  internal mutating func _reverseUntil(_ limit: Index) -> (Index, Index) {
    var f = startIndex
    var l = endIndex
    while f != limit && l != limit {
      formIndex(before: &l)
      swapAt(f, l)
      formIndex(after: &f)
    }
    return (f, l)
  }

  /// Rotates the elements of the collection so that the element
  /// at `middle` ends up first.
  ///
  /// - Returns: The new index of the element that was first
  ///   pre-rotation.
  /// - Complexity: O(*n*)
  @discardableResult
  public mutating func rotate(shiftingToStart middle: Index) -> Index {
    // FIXME: this algorithm should be benchmarked on arrays against
    // the forward Collection algorithm above to prove that it's
    // actually faster.  The other one sometimes does more swaps, but
    // has better locality properties.  Similarly, we've omitted a
    // specialization of rotate for RandomAccessCollection that uses
    // cycles per section 11.4 in "From Mathematics to Generic
    // Programming" by A. Stepanov because it has *much* worse
    // locality properties than either of the other implementations.
    // Benchmarks should be performed for that algorithm too, just to
    // be sure.
    self[..<middle].reverse()
    self[middle...].reverse()
    let (p, q) = _reverseUntil(middle)
    self[p..<q].reverse()
    return middle == p ? q : p
  }
}

/// Returns the greatest common denominator for `m` and `n`.
internal func _gcd(_ m: Int, _ n: Int) -> Int {
  var (m, n) = (m, n)
  while n != 0 {
    let t = m % n
    m = n
    n = t
  }
  return m
}

extension MutableCollection where Self: RandomAccessCollection {

  /// Rotates elements through a cycle, using `sourceForIndex` to generate
  /// the source index for each movement.
  @inline(__always)
  internal mutating func _rotateCycle(
    start: Index,
    sourceOffsetForIndex: (Index) -> Int
  ) {
    let tmp = self[start]
    var i = start
    var j = index(start, offsetBy: sourceOffsetForIndex(start))
    while j != start {
      self[i] = self[j]
      i = j
      j = index(j, offsetBy: sourceOffsetForIndex(j))
    }
    self[i] = tmp
  }

  /// Rotates the elements of the collection so that the element
  /// at `middle` ends up first.
  ///
  /// - Returns: The new index of the element that was first
  ///   pre-rotation.
  /// - Complexity: O(*n*)
  @discardableResult
  public mutating func rotateRandomAccess(
    shiftingToStart middle: Index
  ) -> Index {
    if middle == startIndex { return endIndex }
    if middle == endIndex { return startIndex }

    // The distance to move an element that is moving ->
    let plus = distance(from: startIndex, to: middle)
    // The distance to move an element that is moving <-
    let minus = distance(from: endIndex, to: middle)
    // The new pivot point, aka the destination for the first element
    let pivot = index(startIndex, offsetBy: -minus)

    // If the difference moving forward and backward are relative primes,
    // the entire rotation will be completed in one cycle. Otherwise, repeat
    // cycle, moving the start point forward with each cycle.
    let cycles = _gcd(numericCast(plus), -numericCast(minus))

    for cycle in 1...cycles {
      _rotateCycle(
        start: index(startIndex, offsetBy: numericCast(cycle)),
        sourceOffsetForIndex: { $0 < pivot ? plus : minus })
    }
    return pivot
  }
}

//===--- Concatenation ----------------------------------------------------===//
//===----------------------------------------------------------------------===//

// Concatenation improves on a flattened array or other collection by
// allowing random-access traversal if the underlying collections are
// random-access.

/// A concatenation of two sequences with the same element type.
public struct Concatenation<Base1: Sequence, Base2: Sequence>: Sequence
where Base1.Element == Base2.Element {
  let _base1: Base1
  let _base2: Base2

  init(_base1: Base1, base2: Base2) {
    self._base1 = _base1
    self._base2 = base2
  }

  public struct Iterator: IteratorProtocol {
    var _iterator1: Base1.Iterator
    var _iterator2: Base2.Iterator

    init(_ concatenation: Concatenation) {
      _iterator1 = concatenation._base1.makeIterator()
      _iterator2 = concatenation._base2.makeIterator()
    }

    public mutating func next() -> Base1.Element? {
      return _iterator1.next() ?? _iterator2.next()
    }
  }

  public func makeIterator() -> Iterator {
    Iterator(self)
  }
}

extension Concatenation: Collection where Base1: Collection, Base2: Collection {
  /// A position in a `Concatenation`.
  public struct Index: Comparable {
    internal enum _Representation: Equatable {
      case first(Base1.Index)
      case second(Base2.Index)
    }

    /// Creates a new index into the first underlying collection.
    internal init(first i: Base1.Index) {
      _position = .first(i)
    }

    /// Creates a new index into the second underlying collection.
    internal init(second i: Base2.Index) {
      _position = .second(i)
    }

    internal let _position: _Representation

    public static func < (lhs: Index, rhs: Index) -> Bool {
      switch (lhs._position, rhs._position) {
      case (.first, .second):
        return true
      case (.second, .first):
        return false
      case let (.first(l), .first(r)):
        return l < r
      case let (.second(l), .second(r)):
        return l < r
      }
    }
  }

  public var startIndex: Index {
    // If `_base1` is empty, then `_base2.startIndex` is either a valid position
    // of an element or equal to `_base2.endIndex`.
    return _base1.isEmpty
      ? Index(second: _base2.startIndex)
      : Index(first: _base1.startIndex)
  }

  public var endIndex: Index {
    return Index(second: _base2.endIndex)
  }

  public subscript(i: Index) -> Base1.Element {
    switch i._position {
    case let .first(i):
      return _base1[i]
    case let .second(i):
      return _base2[i]
    }
  }

  public func index(after i: Index) -> Index {
    switch i._position {
    case let .first(i):
      assert(i != _base1.endIndex)
      let next = _base1.index(after: i)
      return next == _base1.endIndex
        ? Index(second: _base2.startIndex)
        : Index(first: next)
    case let .second(i):
      return Index(second: _base2.index(after: i))
    }
  }
}

extension Concatenation: BidirectionalCollection
where Base1: BidirectionalCollection, Base2: BidirectionalCollection {
  public func index(before i: Index) -> Index {
    assert(i != startIndex, "Can't advance before startIndex")
    switch i._position {
    case let .first(i):
      return Index(first: _base1.index(before: i))
    case let .second(i):
      return i == _base2.startIndex
        ? Index(first: _base1.index(before: _base1.endIndex))
        : Index(second: _base2.index(before: i))
    }
  }
}

extension Concatenation: RandomAccessCollection
where Base1: RandomAccessCollection, Base2: RandomAccessCollection {
  public func index(_ i: Index, offsetBy n: Int) -> Index {
    if n == 0 { return i }
    return n > 0 ? _offsetForward(i, by: n) : _offsetBackward(i, by: -n)
  }

  internal func _offsetForward(
    _ i: Index, by n: Int
  ) -> Index {
    switch i._position {
    case let .first(i):
      let d: Int = _base1.distance(from: i, to: _base1.endIndex)
      if n < d {
        return Index(first: _base1.index(i, offsetBy: numericCast(n)))
      } else {
        return Index(
          second: _base2.index(_base2.startIndex, offsetBy: numericCast(n - d)))
      }
    case let .second(i):
      return Index(second: _base2.index(i, offsetBy: numericCast(n)))
    }
  }

  internal func _offsetBackward(
    _ i: Index, by n: Int
  ) -> Index {
    switch i._position {
    case let .first(i):
      return Index(first: _base1.index(i, offsetBy: -numericCast(n)))
    case let .second(i):
      let d: Int = _base2.distance(from: _base2.startIndex, to: i)
      if n <= d {
        return Index(second: _base2.index(i, offsetBy: -numericCast(n)))
      } else {
        return Index(
          first: _base1.index(_base1.endIndex, offsetBy: -numericCast(n - d)))
      }
    }
  }
}

/// Returns a new collection that presents a view onto the elements of the
/// first collection and then the elements of the second collection.
func concatenate<S1: Sequence, S2: Sequence>(
  _ first: S1,
  _ second: S2
) -> Concatenation<S1, S2> where S1.Element == S2.Element {
  return Concatenation(_base1: first, base2: second)
}

extension Sequence {
  func followed<S: Sequence>(by other: S) -> Concatenation<Self, S>
  where Element == S.Element {
    return concatenate(self, other)
  }
}

//===--- RotatedCollection ------------------------------------------------===//
//===----------------------------------------------------------------------===//

/// A rotated view onto a collection.
public struct RotatedCollection<Base: Collection>: Collection {
  let _base: Base
  let _indices: Concatenation<Base.Indices, Base.Indices>

  init(_base: Base, shiftingToStart i: Base.Index) {
    self._base = _base
    self._indices = concatenate(_base.indices[i...], _base.indices[..<i])
  }

  /// A position in a rotated collection.
  public struct Index: Comparable {
    internal let _index: Concatenation<Base.Indices, Base.Indices>.Index

    public static func < (lhs: Index, rhs: Index) -> Bool {
      return lhs._index < rhs._index
    }
  }

  public var startIndex: Index {
    return Index(_index: _indices.startIndex)
  }

  public var endIndex: Index {
    return Index(_index: _indices.endIndex)
  }

  public subscript(i: Index) -> Base.SubSequence.Element {
    return _base[_indices[i._index]]
  }

  public func index(after i: Index) -> Index {
    return Index(_index: _indices.index(after: i._index))
  }

  public func index(_ i: Index, offsetBy n: Int) -> Index {
    return Index(_index: _indices.index(i._index, offsetBy: n))
  }

  public func distance(from start: Index, to end: Index) -> Int {
    return _indices.distance(from: start._index, to: end._index)
  }

  /// The shifted position of the base collection's `startIndex`.
  public var shiftedStartIndex: Index {
    return Index(
      _index: Concatenation<Base.Indices, Base.Indices>.Index(
        second: _indices._base2.startIndex)
    )
  }

  public func rotated(shiftingToStart i: Index) -> RotatedCollection<Base> {
    return RotatedCollection(_base: _base, shiftingToStart: _indices[i._index])
  }
}

extension RotatedCollection: BidirectionalCollection
where Base: BidirectionalCollection {
  public func index(before i: Index) -> Index {
    return Index(_index: _indices.index(before: i._index))
  }
}

extension RotatedCollection: RandomAccessCollection
where Base: RandomAccessCollection {}

extension Collection {
  /// Returns a view of this collection with the elements reordered such the
  /// element at the given position ends up first.
  ///
  /// The subsequence of the collection up to `i` is shifted to after the
  /// subsequence starting at `i`. The order of the elements within each
  /// partition is otherwise unchanged.
  ///
  ///     let a = [10, 20, 30, 40, 50, 60, 70]
  ///     let r = a.rotated(shiftingToStart: 3)
  ///     // r.elementsEqual([40, 50, 60, 70, 10, 20, 30])
  ///
  /// - Parameter i: The position in the collection that should be first in the
  ///   result. `i` must be a valid index of the collection.
  /// - Returns: A rotated view on the elements of this collection, such that
  ///   the element at `i` is first.
  func rotated(shiftingToStart i: Index) -> RotatedCollection<Self> {
    return RotatedCollection(_base: self, shiftingToStart: i)
  }
}

//===--- Stable Partition -------------------------------------------------===//
//===----------------------------------------------------------------------===//

extension MutableCollectionAlgorithms {
  /// Moves all elements satisfying `isSuffixElement` into a suffix of the
  /// collection, preserving their relative order, and returns the start of the
  /// resulting suffix.
  ///
  /// - Complexity: O(n) where n is the number of elements.
  @discardableResult
  mutating func stablePartition(
    isSuffixElement: (Element) throws -> Bool
  ) rethrows -> Index {
    return try stablePartition(count: count, isSuffixElement: isSuffixElement)
  }

  /// Moves all elements satisfying `isSuffixElement` into a suffix of the
  /// collection, preserving their relative order, and returns the start of the
  /// resulting suffix.
  ///
  /// - Complexity: O(n) where n is the number of elements.
  /// - Precondition: `n == self.count`
  fileprivate mutating func stablePartition(
    count n: Int, isSuffixElement: (Element) throws -> Bool
  ) rethrows -> Index {
    if n == 0 { return startIndex }
    if n == 1 {
      return try isSuffixElement(self[startIndex]) ? startIndex : endIndex
    }
    let h = n / 2
    let i = index(startIndex, offsetBy: h)
    let j = try self[..<i].stablePartition(
      count: h, isSuffixElement: isSuffixElement)
    let k = try self[i...].stablePartition(
      count: n - h, isSuffixElement: isSuffixElement)
    return self[j..<k].rotate(shiftingToStart: i)
  }
}

extension Collection {
  func stablyPartitioned(
    isSuffixElement p: (Element) -> Bool
  ) -> [Element] {
    var a = Array(self)
    a.stablePartition(isSuffixElement: p)
    return a
  }
}

extension LazyCollectionProtocol
where Element == Elements.Element, Elements: Collection {
  func stablyPartitioned(
    isSuffixElement p: (Element) -> Bool
  ) -> LazyCollection<[Element]> {
    return elements.stablyPartitioned(isSuffixElement: p).lazy
  }
}

extension Collection {
  /// Returns the index of the first element in the collection
  /// that matches the predicate.
  ///
  /// The collection must already be partitioned according to the
  /// predicate, as if `self.partition(by: predicate)` had already
  /// been called.
  ///
  /// - Efficiency: At most log(N) invocations of `predicate`, where
  ///   N is the length of `self`.  At most log(N) index offsetting
  ///   operations if `self` conforms to `RandomAccessCollection`;
  ///   at most N such operations otherwise.
  func partitionPoint(
    where predicate: (Element) throws -> Bool
  ) rethrows -> Index {
    var n = distance(from: startIndex, to: endIndex)
    var l = startIndex

    while n > 0 {
      let half = n / 2
      let mid = index(l, offsetBy: half)
      if try predicate(self[mid]) {
        n = half
      } else {
        l = index(after: mid)
        n -= half + 1
      }
    }
    return l
  }
}
