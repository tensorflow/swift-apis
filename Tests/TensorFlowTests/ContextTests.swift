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

import Dispatch
import XCTest

@testable import TensorFlow

final class ContextTests: XCTestCase {
  func testDropout() {
    Context.local.learningPhase = .inference
    let dropout = Dropout<Float>(probability: 0.5)
    let x = Tensor<Float>(repeating: 1.0, shape: [5, 5])
    XCTAssertEqual(dropout(x), x)
    withLearningPhase(LearningPhase.inference) {
      XCTAssertEqual(dropout(x), x)
      withLearningPhase(LearningPhase.training) {
        XCTAssertNotEqual(dropout(x), x)
      }
      XCTAssertEqual(dropout(x), x)
    }
    XCTAssertEqual(dropout(x), x)
  }

  func testMultithreadedDropout() {
    let dropout = Dropout<Float>(probability: 0.5)
    let x = Tensor<Float>(repeating: 1.0, shape: [5, 5])
    Context.local.learningPhase = .inference
    DispatchQueue.concurrentPerform(iterations: 10) { i in
      if i.isMultiple(of: 2) {
        XCTAssertEqual(dropout(x), x)
        withLearningPhase(LearningPhase.training) {
          XCTAssertNotEqual(dropout(x), x)
        }
        XCTAssertEqual(dropout(x), x)
      } else {
        XCTAssertEqual(dropout(x), x)
      }
    }
  }

  static var allTests = [
    ("testDropout", testDropout),
    ("testMultithreadedDropout", testMultithreadedDropout),
  ]
}
