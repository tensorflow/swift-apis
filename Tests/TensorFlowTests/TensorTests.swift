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
        XCTAssertEqual(selectValue(true).scalar, 0)
    }

    func testRankGetter() {
        let vector = Tensor<Int32>([1])
        let matrix = Tensor<Float>([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        let ones = Tensor<Int32>(ones: [1, 2, 2, 2, 2, 2, 1])
        let tensor = Tensor<Int32>(shape: [3, 4, 5], scalars: Array(0..<60))
        XCTAssertEqual(vector.rank, 1)
        XCTAssertEqual(matrix.rank, 2)
        XCTAssertEqual(ones.rank, 7)
        XCTAssertEqual(tensor.rank, 3)
    }

    func testShapeGetter() {
        let vector = Tensor<Int32>([1])
        let matrix = Tensor<Float>([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        let ones = Tensor<Int32>(ones: [1, 2, 2, 2, 2, 2, 1])
        let tensor = Tensor<Int32>(shape: [3, 4, 5], scalars: Array(0..<60))
        XCTAssertEqual(vector.shape, [1])
        XCTAssertEqual(matrix.shape, [2, 3])
        XCTAssertEqual(ones.shape, [1, 2, 2, 2, 2, 2, 1])
        XCTAssertEqual(tensor.shape, [3, 4, 5])
    }

    func testTensorShapeDescription() {
        XCTAssertEqual(Tensor<Int32>(ones: [2, 2]).shape.description, "[2, 2]")
        XCTAssertEqual(Tensor(1).shape.description, "[]")
    }
    
    func testEquality() {
        let tensor = Tensor<Float>([0, 1, 2, 3, 4, 5])
        let zeros = Tensor<Float>(zeros: [6])
        
        XCTAssertTrue(tensor == tensor)
        XCTAssertFalse(tensor != tensor)
        XCTAssertFalse(tensor == zeros)
        XCTAssertTrue(tensor != zeros)
        XCTAssertFalse(tensor == tensor.reshaped(to: [2, 3]))
        XCTAssertTrue(tensor != tensor.reshaped(to: [2, 3]))
    }

    static var allTests = [
        ("testSimpleCond", testSimpleCond),
        ("testRankGetter", testRankGetter),
        ("testShapeGetter", testShapeGetter),
        ("testTensorShapeDescription", testTensorShapeDescription),
        ("testEquality", testEquality),
    ]
}
