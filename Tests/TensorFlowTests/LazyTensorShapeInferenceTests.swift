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

extension LazyTensorOperation {
    /// Returns true if the outputs have been materialized.
    var isMaterialized: Bool { outputs != nil }
}

final class LazyTensorShapeInferenceTests : LazyTensorTestCase {
    func testSimpleShapeComputations() {
        let a = Tensor<Float>(shape: [3, 1], scalars: [1.0, 2.0, 3.0])
        let b = Tensor<Float>(shape: [1, 3], scalars: [1.0, 2.0, 3.0])
        let c = Tensor<Float>(shape: [1, 3], scalars: [4.0, 5.0, 6.0])
        let w = a * b
        let wLazyTensorOperation = w._lazyTensor!.lazyTensorOperation!
        let x = w * c
        let xLazyTensorOperation = x._lazyTensor!.lazyTensorOperation!

        // Make sure that `w` and `x` are not materialized.
        XCTAssertFalse(wLazyTensorOperation.isMaterialized)
        XCTAssertFalse(xLazyTensorOperation.isMaterialized)

        // Examine shape of w and confirm no materialization has happened.
        let wShape = w.shape
        XCTAssertEqual(wShape.rank, 2)
        XCTAssertEqual(wShape.dimensions, [3, 3])
        XCTAssertFalse(wLazyTensorOperation.isMaterialized)
        XCTAssertFalse(xLazyTensorOperation.isMaterialized)

        let xShape = x.shape
        XCTAssertEqual(xShape.rank, 2)
        XCTAssertEqual(xShape.dimensions, [3, 3])
        XCTAssertFalse(wLazyTensorOperation.isMaterialized)
        XCTAssertFalse(xLazyTensorOperation.isMaterialized)

        // Trigger materialization.
        let _ = x._rawTensorHandle
        XCTAssertTrue(wLazyTensorOperation.isMaterialized)
        XCTAssertTrue(xLazyTensorOperation.isMaterialized)
    }

    /// Checks scenarios where shapes are computed from input tensors.
    func testShapeComputationsWithInputTensors() {
        let a = Tensor<Float>(shape: [3, 1], scalars: [1.0, 2.0, 3.0])
        let b = a.reshaped(toShape: [1, 3])

        let bLazyTensorOperation = b._lazyTensor!.lazyTensorOperation!
        XCTAssertFalse(bLazyTensorOperation.isMaterialized)

        let bShape = b.shape
        XCTAssertEqual(bShape.rank, 2)
        XCTAssertEqual(bShape.dimensions, [1, 3])
        XCTAssertFalse(bLazyTensorOperation.isMaterialized)

        let c = Tensor<Float>(repeating: 5, shape: [4, 5, 6])
        let cLazyTensorOperation = c._lazyTensor!.lazyTensorOperation!
        XCTAssertFalse(cLazyTensorOperation.isMaterialized)

        let cShape = c.shape
        XCTAssertEqual(cShape.rank, 3)
        XCTAssertEqual(cShape.dimensions, [4, 5, 6])
        XCTAssertFalse(cLazyTensorOperation.isMaterialized)

        // Trigger materialization.
        let _ = b._rawTensorHandle
        let _ = c._rawTensorHandle
        XCTAssertTrue(bLazyTensorOperation.isMaterialized)
        XCTAssertTrue(cLazyTensorOperation.isMaterialized)
    }

    static var allTests = [
        ("testSimpleShapeComputations", testSimpleShapeComputations),
        ("testShapeComputationsWithInputTensors", testShapeComputationsWithInputTensors)
    ]
}

