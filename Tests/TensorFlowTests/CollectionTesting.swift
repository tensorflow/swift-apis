// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

import XCTest

// ************************************************
// Checking the traversal properties of collections.

/// A type that “generic type predicates” can conform to when their result is
/// true.
fileprivate protocol True {}

/// A “generic type predicate” that detects whether a type is a
/// `BidirectionalCollection`.
fileprivate struct IsBidirectionalCollection<C> {}
extension IsBidirectionalCollection: True where C : BidirectionalCollection {  }

/// A “generic type predicate” that detects whether a type is a
/// `RandomAccessCollection`.
fileprivate struct IsRandomAccessCollection<C> {}
extension IsRandomAccessCollection: True where C : RandomAccessCollection {  }

extension Collection {
  /// True iff `Self` conforms to `BidirectionalCollection`.
  ///
  /// Useful in asserting that a certain collection is *not* declared to conform
  /// to `BidirectionalCollection`.
  var isBidirectional: Bool {
    return IsBidirectionalCollection<Self>.self is True.Type
  }

  /// True iff `Self` conforms to `RandomAccessCollection`.
  ///
  /// Useful in asserting that a certain collection is *not* declared to conform
  /// to `RandomAccessCollection`.
  var isRandomAccess: Bool {
    return IsRandomAccessCollection<Self>.self is True.Type
  }
}

// *********************************************************************
// Checking collection semantics.  Note that these checks cannot see any
// declarations that happen to shadow the protocol requirements. Those shadows
// have to be tested separately.

extension Collection where Element: Equatable {
  /// XCTests `self`'s semantic conformance to `Collection`, expecting its
  /// elements to match `expectedValues`.
  ///
  /// - Complexity: O(N²), where N is `self.count`.
  /// - Note: the fact that a call to this method compiles verifies static
  ///   conformance.
  func checkCollectionConformance<
    ExpectedValues: Collection>(expectedValues: ExpectedValues)
  where ExpectedValues.Element == Element
  {
    var i = startIndex
    var firstPassElements: [Element] = []
    var remainingCount: Int = expectedValues.count
    var offset: Int = 0
    var expectedIndices = indices[...]
    
    while i != endIndex {
      XCTAssertEqual(
        i, expectedIndices.popFirst()!,
        "elements of indices don't match index(after:) results.")
      
      XCTAssertLessThan(i, endIndex)
      let j = self.index(after: i)
      XCTAssertLessThan(i, j)
      firstPassElements.append(self[i])
      
      XCTAssertEqual(index(i, offsetBy: remainingCount), endIndex)
      if offset != 0 {
        XCTAssertEqual(
          index(startIndex, offsetBy: offset - 1, limitedBy: i),
          index(startIndex, offsetBy: offset - 1))
      }
      
      XCTAssertEqual(
        index(startIndex, offsetBy: offset, limitedBy: i), i)
      
      if remainingCount != 0 {
        XCTAssertEqual(
          index(startIndex, offsetBy: offset + 1, limitedBy: i), nil)
      }
      
      XCTAssertEqual(distance(from: i, to: endIndex), remainingCount)
      i = j
      remainingCount -= 1
      offset += 1
    }
    XCTAssert(firstPassElements.elementsEqual(expectedValues))
    
    // Check that the second pass has the same elements.  We've verified that
    // indices
    XCTAssert(indices.lazy.map { self[$0] }.elementsEqual(expectedValues))
  }

  /// Returns `index(i, offsetBy: n)`, invoking the implementation that
  /// satisfies the generic requirement, without interference from anything that
  /// happens to shadow it.
  func generic_index(_ i: Index, offsetBy n: Int) -> Index {
    index(i, offsetBy: n)
  }

  /// Returns `index(i, offsetBy: n, limitedBy: limit)`, invoking the
  /// implementation that satisfies the generic requirement, without
  /// interference from anything that happens to shadow it.
  func generic_index(
    _ i: Index, offsetBy n: Int, limitedBy limit: Index
  ) -> Index? {
    index(i, offsetBy: n, limitedBy: limit)
  }

  /// Returns `distance(from: i, to: j)`, invoking the
  /// implementation that satisfies the generic requirement, without
  /// interference from anything that happens to shadow it.
  func generic_distance(from i: Index, to j: Index) -> Int {
    distance(from: i, to: j)
  }
}

extension BidirectionalCollection where Element: Equatable {
  /// XCTests `self`'s semantic conformance to `BidirectionalCollection`,
  /// expecting its elements to match `expectedValues`.
  ///
  /// - Complexity: O(N²), where N is `self.count`.
  /// - Note: the fact that a call to this method compiles verifies static
  ///   conformance.
  func checkBidirectionalCollectionConformance<
    ExpectedValues: Collection>(expectedValues: ExpectedValues)
  where ExpectedValues.Element == Element
  {
    checkCollectionConformance(expectedValues: expectedValues)
    var i = startIndex
    while i != endIndex {
      let j = index(after: i)
      XCTAssertEqual(index(before: j), i)
      let offset = distance(from: i, to: startIndex)
      XCTAssertLessThanOrEqual(offset, 0)
      XCTAssertEqual(index(i, offsetBy: offset), startIndex)
      i = j
    }
  }
}

/// Shared storage for operation counts.
///
/// This is a class:
/// - so that increments aren't missed due to copies
/// - because non-mutating operations on `RandomAccessOperationCounter` have
///   to update it.
class RandomAccessOperationCounts {
  /// The number of invocations of `index(after:)`
  var indexAfter = 0
  /// The number of invocations of `index(before:)`
  var indexBefore = 0

  /// Reset all counts to zero.
  func reset() { (indexAfter, indexBefore) = (0, 0) }
}


/// A wrapper over some `Base` collection that counts index increment/decrement
/// operations.
///
/// This wrapper is useful for verifying that generic collection types that
/// conditionally conform to `RandomAccessCollection` are actually providing the
/// correct complexity.
struct RandomAccessOperationCounter<Base: RandomAccessCollection> {
  var base: Base
  
  typealias Index = Base.Index
  typealias Element = Base.Element

  /// The number of index incrementat/decrement operations applied to `self` and
  /// all its copies.
  var operationCounts = RandomAccessOperationCounts()
}

extension RandomAccessOperationCounter: RandomAccessCollection {  
  var startIndex: Index { base.startIndex }
  var endIndex: Index { base.endIndex }
  subscript(i: Index) -> Base.Element { base[i] }
  
  func index(after i: Index) -> Index {
    operationCounts.indexAfter += 1
    return base.index(after: i)
  }
  func index(before i: Index) -> Index {
    operationCounts.indexBefore += 1
    return base.index(before: i)
  }
  func index(_ i: Index, offsetBy n: Int) -> Index {
    base.index(i, offsetBy: n)
  }

  func index(_ i: Index, offsetBy n: Int, limitedBy limit: Index) -> Index? {
    base.index(i, offsetBy: n, limitedBy: limit)
  }

  func distance(from i: Index, to j: Index) -> Int {
    base.distance(from: i, to: j)
  }
}

extension RandomAccessCollection where Element: Equatable {
  /// XCTests `self`'s semantic conformance to `RandomAccessCollection`,
  /// expecting its elements to match `expectedValues`.
  ///
  /// - Parameter operationCounts: if supplied, should be an instance that
  ///   tracks operations in copies of `self`.
  ///
  /// - Complexity: O(N²), where N is `self.count`.
  ///
  /// - Note: the fact that a call to this method compiles verifies static
  ///   conformance.
  func checkRandomAccessCollectionConformance<ExpectedValues: Collection>(
    expectedValues: ExpectedValues,
    operationCounts: RandomAccessOperationCounts = .init()
  )
  where ExpectedValues.Element == Element
  {
    checkBidirectionalCollectionConformance(expectedValues: expectedValues)
    operationCounts.reset()
    
    XCTAssertEqual(generic_distance(from: startIndex, to: endIndex), count)
    XCTAssertEqual(operationCounts.indexAfter, 0)
    XCTAssertEqual(operationCounts.indexBefore, 0)
    
    XCTAssertEqual(generic_distance(from: endIndex, to: startIndex), -count)
    XCTAssertEqual(operationCounts.indexAfter, 0)
    XCTAssertEqual(operationCounts.indexBefore, 0)

    XCTAssertEqual(index(startIndex, offsetBy: count), endIndex)
    XCTAssertEqual(operationCounts.indexAfter, 0)
    XCTAssertEqual(operationCounts.indexBefore, 0)
    
    XCTAssertEqual(index(endIndex, offsetBy: -count), startIndex)
    XCTAssertEqual(operationCounts.indexAfter, 0)
    XCTAssertEqual(operationCounts.indexBefore, 0)

    XCTAssertEqual(
      index(startIndex, offsetBy: count, limitedBy: endIndex), endIndex)
    XCTAssertEqual(operationCounts.indexAfter, 0)
    XCTAssertEqual(operationCounts.indexBefore, 0)
    
    XCTAssertEqual(
      index(endIndex, offsetBy: -count, limitedBy: startIndex), startIndex)
    XCTAssertEqual(operationCounts.indexAfter, 0)
    XCTAssertEqual(operationCounts.indexBefore, 0)
  }
}
