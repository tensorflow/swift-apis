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

import XCTest

@testable import TensorFlow

final class ComparisonOperatorTests: XCTestCase {
  func testElementwiseComparison() {
    let x = Tensor<Float>([0, 1, 2])
    let y = Tensor<Float>([2, 1, 3])
    XCTAssertEqual((x .< y).scalars, [true, false, true])
  }

  func testLexicographicalComparison() {
    let x = Tensor<Float>([0, 1, 2, 3, 4])
    let y = Tensor<Float>([2, 3, 4, 5, 6])
    XCTAssertTrue((x .< y).all())
  }

  func testIsAlmostEqual() {
    let x = Tensor<Float>([0.1, 0.2, 0.3, 0.4])
    let y = Tensor<Float>([0.0999, 0.20001, 0.2998, 0.4])
    let z = Tensor<Float>([0.0999, 0.20001, 0.2998, 0.3])

    XCTAssertTrue(x.isAlmostEqual(to: y, tolerance: 0.01))
    XCTAssertFalse(x.isAlmostEqual(to: z))

    let nanInf = Tensor<Float>([.nan, .infinity])
    XCTAssertFalse(nanInf.isAlmostEqual(to: nanInf))
  }

  static var allTests = [
    ("testElementwiseComparison", testElementwiseComparison),
    ("testLexicographicalComparison", testLexicographicalComparison),
    ("testIsAlmostEqual", testIsAlmostEqual),
  ]
}
