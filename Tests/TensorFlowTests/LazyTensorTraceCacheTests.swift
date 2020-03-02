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

import CTensorFlow
import XCTest

@testable import TensorFlow

final class LazyTensorTraceCacheTests: LazyTensorTestCase {
  override class func setUp() {
    super.setUp()
    LazyTensorContext.local.shouldPromoteConstants = true
  }

  override class func tearDown() {
    super.tearDown()
    LazyTensorTraceCache.clearCache()
  }

  func testConstPromotion() {
    LazyTensorTraceCache.clearCache()
    let a = Tensor<Float>(1.0)
    let b = Tensor<Float>(2.0)
    let c = Tensor<Float>(3.0)
    let d = Tensor<Float>(4.0)
    let w = a * b
    let x = c * d
    // Trigger materialization for `w` so that a trace with constants and mul is added to cache.
    XCTAssertEqual(
      lazyTrace(w).description,
      """
      lazyTrace_3() -> (%2) {
        %0 = Const[dtype: float, value: 1.0]()
        %1 = Const[dtype: float, value: 2.0]()
        %2 = Mul[T: float](%0, %1)
      }
      """)
    XCTAssertEqual(w.scalars, [2.0])

    // The trace for `x` should have the inputs to Mul as arguments instead of constants.
    XCTAssertEqual(
      lazyTrace(x).description,
      """
      lazyTrace_3(%0: float, %1: float) -> (%2) {
        %2 = Mul[T: float](%0, %1)
      }
      """)
    XCTAssertEqual(x.scalarized(), 12.0)

    let e = Tensor<Float>(shape: [1, 3], scalars: [1, 2, 3])
    let f = Tensor<Float>(5.0)
    let y = e * f
    // We won't promote constants in 'y' as shape of constants is different.
    XCTAssertEqual(
      lazyTrace(y).description,
      """
      lazyTrace_3() -> (%2) {
        %0 = Const[dtype: float, value: [[1.0, 2.0, 3.0]]]()
        %1 = Const[dtype: float, value: 5.0]()
        %2 = Mul[T: float](%0, %1)
      }
      """)
    XCTAssertEqual(y.scalars, [5.0, 10.0, 15.0])
  }

  func testDoNotPromoteEqualConstants() {
    LazyTensorTraceCache.clearCache()
    let a = Tensor<Float>(1.0)
    let b = Tensor<Float>(2.0)
    let c = Tensor<Float>(3.0)
    let w = a * b
    let x = a * c
    XCTAssertEqual(
      lazyTrace(w).description,
      """
      lazyTrace_3() -> (%2) {
        %0 = Const[dtype: float, value: 1.0]()
        %1 = Const[dtype: float, value: 2.0]()
        %2 = Mul[T: float](%0, %1)
      }
      """)
    XCTAssertEqual(w.scalars, [2.0])
    // Const 1.0 is not promoted.
    XCTAssertEqual(
      lazyTrace(x).description,
      """
      lazyTrace_3(%1: float) -> (%2) {
        %0 = Const[dtype: float, value: 1.0]()
        %2 = Mul[T: float](%0, %1)
      }
      """)
  }

  private func lazyTensorOperation<T: TensorFlowScalar>(
    _ input: Tensor<T>
  ) -> LazyTensorOperation? {
    let tensor = input.handle.handle
    guard let lazyTensor = tensor as? LazyTensorHandle else {
      XCTFail("Trying to get lazy trace for a non-lazy tensor.")
      return nil
    }
    guard case let .symbolic(lazyOp, _, _) = lazyTensor.handle else {
      XCTFail("Cannot get lazy trace for a concrete tensor.")
      return nil
    }
    return lazyOp
  }

  private func lazyTrace<T: TensorFlowScalar>(
    _ input: Tensor<T>
  ) -> LazyTensorTrace {
    let lazyOperation = lazyTensorOperation(input)!
    return LazyTensorTraceBuilder.materializationTraceInfo(lazyOperation).trace
  }

  static var allTests = [
    ("testConstPromotion", testConstPromotion),
    ("testDoNotPromoteEqualConstants", testDoNotPromoteEqualConstants),
  ]
}
