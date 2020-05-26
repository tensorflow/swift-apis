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

import TensorFlow
import XCTest

extension Collection where Element: Equatable {
  /// Tests `self`'s dynamic conformance to `Collection`, expecting its elements
  /// to match `expectedValues`.
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

  func generic_index(_ i: Index, offsetBy n: Int) -> Index {
    index(i, offsetBy: n)
  }

  func generic_index(
    _ i: Index, offsetBy n: Int, limitedBy limit: Index
  ) -> Index? {
    index(i, offsetBy: n, limitedBy: limit)
  }

  func generic_distance(from i: Index, to j: Index) -> Int {
    distance(from: i, to: j)
  }
}

extension BidirectionalCollection where Element: Equatable {
  /// Tests `self`'s dynamic conformance to `BidirectionalCollection`, expecting
  /// its elements to match `expectedValues`.
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

struct RandomAccessOperationCounter<Base: RandomAccessCollection> {
  var base: Base
  
  typealias Index = Base.Index
  typealias Element = Base.Element

  class OperationCounts {
    var indexAfter = 0
    var indexBefore = 0
  }
  
  var operationCounts = OperationCounts()
  
  mutating func reset() {
    operationCounts.indexAfter = 0
    operationCounts.indexBefore = 0
  }
}

extension RandomAccessOperationCounter: Collection {  
  var startIndex: Index { base.startIndex }
  var endIndex: Index { base.endIndex }
  subscript(i: Index) -> Base.Element { base[i] }
  
  func index(after i: Index) -> Index {
    operationCounts.indexAfter += 1
    return base.index(after: i)
  }
}

extension RandomAccessOperationCounter: BidirectionalCollection {
  func index(before i: Index) -> Index {
    operationCounts.indexBefore += 1
    return base.index(before: i)
  }
}

extension RandomAccessOperationCounter: RandomAccessCollection {
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

final class SamplingTests: XCTestCase {
  func test_init() {
    let a = Sampling(base: Array(3..<100), selection: [7, 9, 8])
    XCTAssert(a.elementsEqual([10, 12, 11]))

    let b = Set(0...20)
    let c = Sampling(base: b, selection: b.indices)
    XCTAssert(b.elementsEqual(c))

    let d = Sampling(
      base: b,
      selection: (0...20).lazy.map { b.firstIndex(of: $0)! })
    
    XCTAssert(d.elementsEqual(0...20))
  }

  func test_Collection() {
    let b = 0...20
    let d = Sampling(base: b, selection: AnyCollection(b.indices))
    d.checkCollectionConformance(expectedValues: b)
  }

  func test_BidirectionalCollection() {
    let b = 0...20
    let d = Sampling(
      base: b, selection: AnyBidirectionalCollection(b.indices))
    d.checkBidirectionalCollectionConformance(expectedValues: b)
  }

  func test_RandomAccessCollection() {
    let b = 0...20
    let d = Sampling(base: b, selection: b.indices)
    d.checkBidirectionalCollectionConformance(expectedValues: b)
    
    let instrumentedIndices = RandomAccessOperationCounter(base: b.indices)
    let operationCounts = instrumentedIndices.operationCounts
    let d1 = Sampling(base: b, selection: instrumentedIndices)

    XCTAssertEqual(
      d1.generic_distance(from: d1.startIndex, to: d1.endIndex),
      d1.count)
    XCTAssertEqual(operationCounts.indexAfter, 0)
    XCTAssertEqual(operationCounts.indexBefore, 0)
    
    XCTAssertEqual(
      d1.generic_distance(from: d1.endIndex, to: d1.startIndex),
      -d1.count)
    XCTAssertEqual(operationCounts.indexAfter, 0)
    XCTAssertEqual(operationCounts.indexBefore, 0)

    XCTAssertEqual(d1.index(d1.startIndex, offsetBy: d1.count), d1.endIndex)
    XCTAssertEqual(operationCounts.indexAfter, 0)
    XCTAssertEqual(operationCounts.indexBefore, 0)
    
    XCTAssertEqual(d1.index(d1.endIndex, offsetBy: -d1.count), d1.startIndex)
    XCTAssertEqual(operationCounts.indexAfter, 0)
    XCTAssertEqual(operationCounts.indexBefore, 0)

    XCTAssertEqual(
      d1.index(d1.startIndex, offsetBy: d1.count, limitedBy: d1.endIndex),
      d1.endIndex)
    XCTAssertEqual(operationCounts.indexAfter, 0)
    XCTAssertEqual(operationCounts.indexBefore, 0)
    
    XCTAssertEqual(
      d1.index(d1.endIndex, offsetBy: -d1.count, limitedBy: d1.startIndex),
      d1.startIndex)
    XCTAssertEqual(operationCounts.indexAfter, 0)
    XCTAssertEqual(operationCounts.indexBefore, 0)
}

  static var allTests = [
    ("test_init", test_init),
    ("test_Collection", test_Collection),
    ("test_BidirectionalCollection", test_BidirectionalCollection),
    ("test_RandomAccessCollection", test_RandomAccessCollection),
  ]
}
