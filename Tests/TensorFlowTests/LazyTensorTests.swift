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

struct LazyTensorOperationRef: Equatable, Hashable {
    let value: LazyTensorOperation
    init(_ value: LazyTensorOperation) { self.value = value }
    func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(value))
    }
}

func ==(lhs: LazyTensorOperationRef, rhs: LazyTensorOperationRef) -> Bool {
    return lhs.value === rhs.value
}

final class LazyTensorTests: XCTestCase {
    func testConstructions() {
        let zero = Tensor<Float>(0.0)
        let zeroTFEHandle = zero.handle.handle._tfeTensorHandle

        let concTensor = LazyTensor(zeroTFEHandle)
        XCTAssertEqual(concTensor.description, "0.0")

        let materializedConcTensor = LazyTensor(
            _materialized: zeroTFEHandle)
        XCTAssertEqual(materializedConcTensor.description, "0.0*")

        let op = LazyTensorOperation(
            _id: "0", name: "IdentityN", outputCount: 3)
        let symTensor0 = LazyTensor(_lazy: op, index: 0)
        XCTAssertEqual(symTensor0.description, "IdentityN_0:0")

        let symTensor1 = LazyTensor(_lazy: op, index: 2)
        XCTAssertEqual(symTensor1.description, "IdentityN_0:2")

        let liveSymTensor = LazyTensor(_lazyLive: op, index: 0)
        XCTAssertEqual(liveSymTensor.description, "IdentityN_0:0*")
    }

    func testLivenessTracking() {
        func assertLive(_ expectedLive: [LazyTensorOperation]) {
            var actualLiveOps: Set<LazyTensorOperationRef> = []
            LazyTensor.forEachLiveOperation {
                actualLiveOps.insert(LazyTensorOperationRef($0))
            }
            let expectedLiveOps = Set<LazyTensorOperationRef>(
                expectedLive.map { LazyTensorOperationRef($0) }
            )
            XCTAssertEqual(expectedLiveOps, actualLiveOps)
        }

        func assertAll(_ expectedAll: [LazyTensorOperation]) {
            var actualAllOps: Set<LazyTensorOperationRef> = []
            LazyTensor.forEachOperation {
                actualAllOps.insert(LazyTensorOperationRef($0))
            }
            let expectedAllOps = Set<LazyTensorOperationRef>(
                expectedAll.map { LazyTensorOperationRef($0) }
            )
            XCTAssertEqual(expectedAllOps, actualAllOps)
        }

        func isSymbolic(_ t: LazyTensor) -> Bool {
            if case let .symbolic(_) = t.handle {
                return true
            } else {
                return false
            }
        }

        let op0 = LazyTensorOperation(
            _id: "0", name: "IdentityN", outputCount: 2)
        let op1 = LazyTensorOperation(
            _id: "1", name: "IdentityN", outputCount: 2)

        XCTAssertFalse(LazyTensor.isLive(op0))
        XCTAssertFalse(LazyTensor.isLive(op1))

        let t0 = LazyTensor(_lazyLive: op0, index: 0)
        let t1 = LazyTensor(_lazy: op1, index: 1)
        XCTAssertTrue(LazyTensor.isLive(op0))
        XCTAssertFalse(LazyTensor.isLive(op1))

        do {
            let t3 = LazyTensor(_lazyLive: op1, index: 0)
            XCTAssertTrue(LazyTensor.isLive(op1))
            assertLive([op0, op1])
            assertAll([op0, op1])
            // The following is here just to ensure t3 is live.
            XCTAssertTrue(isSymbolic(t3))
        }
        XCTAssertFalse(LazyTensor.isLive(op1))
        assertLive([op0])
        assertAll([op0, op1])

        // The following are here just to ensure t0 and t1 are live.
        XCTAssertTrue(isSymbolic(t1))
        XCTAssertTrue(isSymbolic(t0))
    }

    static var allTests = [
        ("testConstructions", testConstructions),
        ("testLivenessTracking", testLivenessTracking),
    ]

}
