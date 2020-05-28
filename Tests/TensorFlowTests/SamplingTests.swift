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
    XCTAssertFalse(d.isBidirectional)
    XCTAssertFalse(d.isRandomAccess)
  }

  func test_BidirectionalCollection() {
    let b = 0...20
    let d = Sampling(
      base: b, selection: AnyBidirectionalCollection(b.indices))
    d.checkBidirectionalCollectionConformance(expectedValues: b)
    XCTAssert(d.isBidirectional)
    XCTAssertFalse(d.isRandomAccess)
  }

  func test_RandomAccessCollection() {
    // Can't use 0...20 directly because of
    // https://bugs.swift.org/browse/SR-1288.  Array's Indices seem to work
    // properly, though.
    let b = Array(0...20)
    let d = Sampling(base: b, selection: b.indices)
    d.checkRandomAccessCollectionConformance(expectedValues: b)
    XCTAssert(d.isBidirectional)
    XCTAssert(d.isRandomAccess)
    
    let instrumentedIndices = RandomAccessOperationCounter(base: b.indices)
    let d1 = Sampling(base: b, selection: instrumentedIndices)
    d1.checkRandomAccessCollectionConformance(
      expectedValues: b, operationCounts: instrumentedIndices.operationCounts)
}

  static var allTests = [
    ("test_init", test_init),
    ("test_Collection", test_Collection),
    ("test_BidirectionalCollection", test_BidirectionalCollection),
    ("test_RandomAccessCollection", test_RandomAccessCollection),
  ]
}
