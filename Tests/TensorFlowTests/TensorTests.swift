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

final class TensorTests: XCTestCase {
    func testSimpleCond() {
        func selectValue(_ pred: Bool) -> Tensor<Int32> {
            let a = Tensor<Int32>(0)
            let b = Tensor<Int32>(1)
            if pred {
                return a
            }
            return b
        }
        XCTAssertEqual(0, selectValue(true).scalar)
    }

    func testRankGetter() {
        let vector = Tensor<Int32>([1])
        let matrix = Tensor<Float>([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        let ones = Tensor<Int32>(ones: [1, 2, 2, 2, 2, 2, 1])
        let tensor = Tensor<Int32>(shape: [3, 4, 5], scalars: Array(0..<60))
        XCTAssertEqual(1, vector.rank)
        XCTAssertEqual(2, matrix.rank)
        XCTAssertEqual(7, ones.rank)
        XCTAssertEqual(3, tensor.rank)
    }

    func testShapeGetter() {
        let vector = Tensor<Int32>([1])
        let matrix = Tensor<Float>([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        let ones = Tensor<Int32>(ones: [1, 2, 2, 2, 2, 2, 1])
        let tensor = Tensor<Int32>(shape: [3, 4, 5], scalars: Array(0..<60))
        XCTAssertEqual([1], vector.shape)
        XCTAssertEqual([2, 3], matrix.shape)
        XCTAssertEqual([1, 2, 2, 2, 2, 2, 1], ones.shape)
        XCTAssertEqual([3, 4, 5], tensor.shape)
    }

    func testTensorShapeDescription() {
        XCTAssertEqual("[2, 2]", Tensor<Int32>(ones: [2, 2]).shape.description)
        XCTAssertEqual("[]", Tensor(1).shape.description)
    }

    static var allTests = [
        ("testSimpleCond", testSimpleCond),
        ("testRankGetter", testRankGetter),
        ("testShapeGetter", testShapeGetter),
        ("testTensorShapeDescription", testTensorShapeDescription)
    ]
}
