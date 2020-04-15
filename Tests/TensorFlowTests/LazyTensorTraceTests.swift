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

final class LazyTensorTraceTests: LazyTensorTestCase {
  func testSingleLiveTensor() {
    let a = Tensor<Float>(10.0)
    let b = Tensor<Float>(2.0)
    let c = Tensor<Float>(3.0)
    let w = a + b * c
    XCTAssertEqual(
      lazyTrace(w).description,
      """
      lazyTrace_5() -> (%4) {
        %0 = Const[dtype: float, value: 10.0]()
        %1 = Const[dtype: float, value: 2.0]()
        %2 = Const[dtype: float, value: 3.0]()
        %3 = Mul[T: float](%1, %2)
        %4 = AddV2[T: float](%0, %3)
      }
      """)
  }

  func testMultipleLiveTensors() {
    // This test checks that *only* the operations that correspond to `w`,
    // `y` and `z` are marked as outputs. Specifcally, the intermediate
    // operations in the trace are not marked as outputs.
    let a = Tensor<Float>(10.0)
    let b = Tensor<Float>(2.0)
    let c = Tensor<Float>(3.0)
    let w = a + b + c
    let y = w * c
    let z = y / (w - c)
    XCTAssertEqual(
      lazyTrace(z).description,
      """
      lazyTrace_8() -> (%4, %5, %7) {
        %0 = Const[dtype: float, value: 10.0]()
        %1 = Const[dtype: float, value: 2.0]()
        %2 = AddV2[T: float](%0, %1)
        %3 = Const[dtype: float, value: 3.0]()
        %4 = AddV2[T: float](%2, %3)
        %5 = Mul[T: float](%4, %3)
        %6 = Sub[T: float](%4, %3)
        %7 = Div[T: float](%5, %6)
      }
      """)

    // Note that we only pick operations on which the lazy tensor in
    // question depends on.
    XCTAssertEqual(
      lazyTrace(y).description,
      """
      lazyTrace_6() -> (%4, %5) {
        %0 = Const[dtype: float, value: 10.0]()
        %1 = Const[dtype: float, value: 2.0]()
        %2 = AddV2[T: float](%0, %1)
        %3 = Const[dtype: float, value: 3.0]()
        %4 = AddV2[T: float](%2, %3)
        %5 = Mul[T: float](%4, %3)
      }
      """)
  }

  func testMultipleTargets() {
    let a = Tensor<Float>(1.0)
    let b = Tensor<Float>(2.0)
    let c = Tensor<Float>(3.0)
    let d = Tensor<Float>(4.0)
    let w = a + b
    let x = c + d
    let lazyOps = [w, x].map { self.lazyTensorOperation($0)! }
    XCTAssertEqual(
      lazyTrace(lazyOps).description,
      """
      lazyTrace_6() -> (%2, %5) {
        %0 = Const[dtype: float, value: 1.0]()
        %1 = Const[dtype: float, value: 2.0]()
        %2 = AddV2[T: float](%0, %1)
        %3 = Const[dtype: float, value: 3.0]()
        %4 = Const[dtype: float, value: 4.0]()
        %5 = AddV2[T: float](%3, %4)
      }
      """)
  }

  func testSimpleControlFlow() {
    let a = Tensor<Float>(5.0)
    let addOrMul = { (useAdd: Bool, a: Tensor<Float>) in
      useAdd ? (a + a) : (a * a)
    }
    let add = addOrMul( /*useAdd:*/true, a)
    XCTAssertEqual(
      lazyTrace(add).description,
      """
      lazyTrace_2() -> (%1) {
        %0 = Const[dtype: float, value: 5.0]()
        %1 = AddV2[T: float](%0, %0)
      }
      """)
    let mul = addOrMul( /*useAdd:*/false, a)
    XCTAssertEqual(
      lazyTrace(mul).description,
      """
      lazyTrace_2() -> (%1) {
        %0 = Const[dtype: float, value: 5.0]()
        %1 = Mul[T: float](%0, %0)
      }
      """)
  }

  func testManualConstPromotion() {
    let a = Tensor<Float>(10.0)
    let b = Tensor<Float>(2.0)

    // Since `lazyA` is not marked as an input, this will
    // be burnt into the trace as a constant.
    let lazyA = a._concreteLazyTensor
    let w1 = lazyA * b
    let w1LazyOp = lazyTensorOperation(w1)!
    let w1TraceInfo = LazyTensorTraceBuilder.materializationTraceInfo(w1LazyOp)
    let w1Trace = w1TraceInfo.trace
    XCTAssertEqual(
      w1Trace.description,
      """
      lazyTrace_3() -> (%2) {
        %0 = Const[dtype: float, value: 10.0]()
        %1 = Const[dtype: float, value: 2.0]()
        %2 = Mul[T: float](%0, %1)
      }
      """)
    XCTAssertEqual(w1TraceInfo.concreteInputs.count, 0)

    // Since `lazyInputA` is marked as an input, this will
    // be promoted to an input for the trace.
    let inputLazyA = a._concreteInputLazyTensor
    let w2 = inputLazyA * b
    let w2LazyOp = lazyTensorOperation(w2)!
    let w2TraceInfo = LazyTensorTraceBuilder.materializationTraceInfo(w2LazyOp)
    let w2Trace = w2TraceInfo.trace
    XCTAssertEqual(
      w2Trace.description,
      """
      lazyTrace_3(%0: float) -> (%2) {
        %1 = Const[dtype: float, value: 2.0]()
        %2 = Mul[T: float](%0, %1)
      }
      """)
    // Make sure that the promoted constants are gathered as `inputValues`.
    XCTAssertEqual(w2TraceInfo.concreteInputs.count, 1)
    XCTAssertEqual(w2TraceInfo.concreteInputs[0].valueDescription, "10.0")
  }

  func testConstPromotion() {
    let a = Tensor<Float>(1.0)
    let b = Tensor<Float>(2.0)
    let c = Tensor<Float>(3.0)
    let y = a + b
    let z = y * c

    XCTAssertEqual(
      lazyTrace(y).description,
      """
      lazyTrace_3() -> (%2) {
        %0 = Const[dtype: float, value: 1.0]()
        %1 = Const[dtype: float, value: 2.0]()
        %2 = AddV2[T: float](%0, %1)
      }
      """)
    XCTAssertEqual(y.scalarized(), 3.0)

    /// Now that `y` is materialized and a constant,
    /// the trace for `z` will use that as a constant.
    let zLazyOp = lazyTensorOperation(z)!
    let zTraceInfo = LazyTensorTraceBuilder.materializationTraceInfo(zLazyOp)
    let zTrace = zTraceInfo.trace
    XCTAssertEqual(
      zTrace.description,
      """
      lazyTrace_3(%0: float) -> (%2) {
        %1 = Const[dtype: float, value: 3.0]()
        %2 = Mul[T: float](%0, %1)
      }
      """)
    // Make sure that the promoted constants are gathered as `inputValues`.
    XCTAssertEqual(zTraceInfo.concreteInputs.count, 1)
    XCTAssertEqual(zTraceInfo.concreteInputs[0].valueDescription, "3.0")
    XCTAssertEqual(z.scalarized(), 9.0)
  }

  func testTraceWithFunctionAttributes() {
    typealias Int32Pair = Zip2TensorGroup<Tensor<Int32>, Tensor<Int32>>
    func thenBranch(x: Tensor<Float>) -> Tensor<Float> {
      return x + 10.0
    }
    func elseBranch(x: Tensor<Float>) -> Tensor<Float> {
      return x - 9.0
    }
    let c: Tensor<Float> = _Raw.if_(
      cond: Tensor<Bool>(false),
      Tensor<Float>(20.0),
      thenBranch: thenBranch,
      elseBranch: elseBranch,
      outputShapes: [nil])
    let cLazyOp = lazyTensorOperation(c)!
    let cTraceInfo = LazyTensorTraceBuilder.materializationTraceInfo(cLazyOp)
    let cTrace = cTraceInfo.trace
    XCTAssertEqual(
      cTrace.description,
      """
      lazyTrace_3() -> (%2) {
        %0 = Const[dtype: bool, value: false]()
        %1 = Const[dtype: float, value: 20.0]()
        %2 = If[Tcond: bool, Tin: [float], Tout: [float], else_branch: TFFunction(lazyTrace_3_kMDsaAFRUp8), output_shapes: [nil], then_branch: TFFunction(lazyTrace_3_sayLTaDTeLE)](%0, [%1])
      }
      """)
    // Returns the result of the else branch.
    XCTAssertEqual(c.scalarized(), 11.0)
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

  private func lazyTrace<T: TensorFlowScalar>(_ input: Tensor<T>) -> LazyTensorTrace {
    let lazyOperation = lazyTensorOperation(input)!
    return lazyTrace([lazyOperation])
  }

  private func lazyTrace(_ lazyOperations: [LazyTensorOperation]) -> LazyTensorTrace {
    return LazyTensorTraceBuilder.materializationTraceInfo(lazyOperations).trace
  }

  static var allTests = [
    ("testSingleLiveTensor", testSingleLiveTensor),
    ("testMultipleLiveTensors", testMultipleLiveTensors),
    ("testMultipleTargets", testMultipleTargets),
    ("testSimpleControlFlow", testSimpleControlFlow),
    ("testManualConstPromotion", testManualConstPromotion),
    ("testConstPromotion", testConstPromotion),
    ("testTraceWithFunctionAttributes", testTraceWithFunctionAttributes),
  ]
}
