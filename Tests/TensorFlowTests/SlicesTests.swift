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

final class SlicesTests: XCTestCase {
  func test_init() {
    let a = 0..<97
    let batchSize = 17
    let b = Slices(0..<97, batchSize: batchSize)
    XCTAssert(b.joined().elementsEqual(a))
    XCTAssert(b.dropLast().allSatisfy { $0.count == 17 })
    XCTAssertEqual(
      b[b.index(b.startIndex, offsetBy: a.count / batchSize)].count,
      a.count % batchSize)
  }
  
  func test_Collection() {
    let a = 0..<97
    let batchSize = 17
    let b = Slices(0..<97, batchSize: batchSize)
    
    var expected: [Range<Int>.SubSequence] = []
    var source = a[...]
    while !source.isEmpty {
      expected.append(source.prefix(batchSize))
      source = source.dropFirst(batchSize)
    }
    b.checkCollectionConformance(expectedValues: expected)
    XCTAssertFalse(b.isBidirectional)
    XCTAssertFalse(b.isRandomAccess)
  }

  static var allTests = [
    ("test_init", test_init),
    ("test_Collection", test_Collection),
  ]
}
