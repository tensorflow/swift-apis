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
import CTensorFlow

final class LazyTensorEvaluationTests: XCTestCase {
    override class func setUp() {
        super.setUp()
        _RuntimeConfig.useLazyTensor = true
    }

    override class func tearDown() {
        super.tearDown()
        _RuntimeConfig.useLazyTensor = false
    }

    func testSimpleOperations() {
        let a = Tensor<Float>(10.0)
        let b = Tensor<Float>(2.0)
        let c = Tensor<Float>(3.0)
        let w = a + b * c

        XCTAssertFalse(isMaterialized(w))
        XCTAssertEqual(w.scalarized(), 16.0)
        XCTAssertTrue(isMaterialized(w))
    }

    func testMultipleMaterializations() {
        let a = Tensor<Float>(10.0)
        let b = Tensor<Float>(2.0)
        let c = Tensor<Float>(3.0)
        let x = a + b + c
        let y = x * c
        let z = y / (x - c)

        // Materialize y first
        XCTAssertFalse(isMaterialized(x))
        XCTAssertFalse(isMaterialized(y))
        XCTAssertFalse(isMaterialized(z))
        XCTAssertEqual(y.scalarized(), 45.0)

        // x and y are materialized, but not z.
        XCTAssertTrue(isMaterialized(x))
        XCTAssertTrue(isMaterialized(y))
        XCTAssertFalse(isMaterialized(z))

        XCTAssertEqual(z.scalarized(), 3.75)
        XCTAssertTrue(isMaterialized(z))
    }

    func testSimpleControlFlow() {
        let a = Tensor<Float>(5.0)
        let addOrMul = { (useAdd: Bool, a: Tensor<Float>) in
            useAdd ? (a + a) : (a * a)
        }
        let add = addOrMul(/*useAdd:*/true, a)
        XCTAssertFalse(isMaterialized(add))
        XCTAssertEqual(add.scalarized(), 10.0);
        XCTAssertTrue(isMaterialized(add))

        let mul = addOrMul(/*useAdd:*/false, a)
        XCTAssertFalse(isMaterialized(mul))
        XCTAssertEqual(mul.scalarized(), 25.0);
        XCTAssertTrue(isMaterialized(mul))
    }

    func testSimpleLoop() {
        var sum = Tensor<Float>(0)
        for i in 1...10 { sum += Float(i) }
        XCTAssertFalse(isMaterialized(sum))
        XCTAssertEqual(sum.scalarized(), 55.0, accuracy: 0.00001)
        XCTAssertTrue(isMaterialized(sum))
    }

    private func isMaterialized<T: TensorFlowScalar>(_ input: Tensor<T>) -> Bool {
        let tensor = input.handle.handle
        guard let lazyTensor = tensor as? LazyTensor else { return true }
        switch lazyTensor.handle {
        case .symbolic(let op, _, _): return op.outputs != nil
        default: return false
        }
    }

    static var allTests = [
        ("testSimpleOperations", testSimpleOperations),
        ("testMultipleMaterializations", testMultipleMaterializations),
        ("testSimpleControlFlow", testSimpleControlFlow),
        ("testSimpleLoop", testSimpleLoop),
    ]
}

