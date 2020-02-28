// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 1)
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

// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 16)

import XCTest
import TensorFlow

struct AddOne: ParameterlessLayer {
    @differentiable
    func callAsFunction(_ input: Float) -> Float {
        return input + 1
    }
}

final class SequencedTests: XCTestCase {
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 29)
    func testSequenced2() {
        let input = Float(0)
	let output = input.sequenced(
	    through: AddOne(), AddOne())
	XCTAssertEqual(output, 2)
    }
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 29)
    func testSequenced3() {
        let input = Float(0)
	let output = input.sequenced(
	    through: AddOne(), AddOne(), AddOne())
	XCTAssertEqual(output, 3)
    }
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 29)
    func testSequenced4() {
        let input = Float(0)
	let output = input.sequenced(
	    through: AddOne(), AddOne(), AddOne(), AddOne())
	XCTAssertEqual(output, 4)
    }
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 29)
    func testSequenced5() {
        let input = Float(0)
	let output = input.sequenced(
	    through: AddOne(), AddOne(), AddOne(), AddOne(), AddOne())
	XCTAssertEqual(output, 5)
    }
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 36)

    static var allTests = [
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 39)
        ("testSequenced2", testSequenced2),
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 39)
        ("testSequenced3", testSequenced3),
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 39)
        ("testSequenced4", testSequenced4),
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 39)
        ("testSequenced5", testSequenced5),
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 41)
    ]
}
