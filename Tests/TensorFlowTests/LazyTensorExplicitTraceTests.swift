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

final class LazyTensorExplicitTraceTests: LazyTensorTestCase {
  func testSingleInput() {
    func fn(x: Tensor<Float>) -> Tensor<Float> { return x + x }
    let trace = LazyTensorTraceBuilder.trace(fn)
    XCTAssertEqual(
      trace.description,
      """
      lazyTrace_2(%0: float) -> (%1) {
        %1 = AddV2[T: float](%0, %0)
      }
      """)
    let outputs = runTrace(trace: trace, input: Tensor<Float>(10.0))
    XCTAssertEqual(outputs.count, 1)
    XCTAssertEqual(outputs[0].valueDescription, "20.0")
  }

  func testTensorGroupInputOutputs() {
    typealias TensorFloatInt32Pair = Zip2TensorGroup<Tensor<Float>, Tensor<Int32>>
    typealias TensorInt32FloatPair = Zip2TensorGroup<Tensor<Int32>, Tensor<Float>>
    func fn(input: TensorFloatInt32Pair) -> TensorInt32FloatPair {
      return TensorInt32FloatPair(input.second * 4, input.first + 3.0)
    }
    let trace = LazyTensorTraceBuilder.trace(fn)
    XCTAssertEqual(
      trace.description,
      """
      lazyTrace_6(%0: float, %1: int32) -> (%3, %5) {
        %2 = Const[dtype: int32, value: 4]()
        %3 = Mul[T: int32](%1, %2)
        %4 = Const[dtype: float, value: 3.0]()
        %5 = AddV2[T: float](%0, %4)
      }
      """)
    let outputs = runTrace(
      trace: trace,
      input: TensorFloatInt32Pair(Tensor<Float>(10.0), Tensor<Int32>(5)))
    XCTAssertEqual(outputs.count, 2)
    XCTAssertEqual(outputs[0].valueDescription, "20")
    XCTAssertEqual(outputs[1].valueDescription, "13.0")
  }

  func testClosureCapturesOfTensors() {
    let x = Tensor<Float>(10.0)
    let y = x + x
    func fn(input: Tensor<Float>) -> Tensor<Float> {
      return input * y
    }
    let trace = LazyTensorTraceBuilder.trace(fn)
    /// Note that the computation x + x is encoded in the trace.
    XCTAssertEqual(
      trace.description,
      """
      lazyTrace_4(%0: float) -> (%3) {
        %1 = Const[dtype: float, value: 10.0]()
        %2 = AddV2[T: float](%1, %1)
        %3 = Mul[T: float](%0, %2)
      }
      """)
    let outputs = runTrace(
      trace: trace,
      input: Tensor<Float>(5.0))
    XCTAssertEqual(outputs.count, 1)
    XCTAssertEqual(outputs[0].valueDescription, "100.0")
  }

  func testClosureCapturesOfNonTensors() {
    let x: Float = 5.0
    func fn(input: Tensor<Float>) -> Tensor<Float> {
      return input * Tensor<Float>(x)
    }
    let trace = LazyTensorTraceBuilder.trace(fn)
    /// Note that the computation x + x is encoded in the trace.
    XCTAssertEqual(
      trace.description,
      """
      lazyTrace_3(%0: float) -> (%2) {
        %1 = Const[dtype: float, value: 5.0]()
        %2 = Mul[T: float](%0, %1)
      }
      """)
    let outputs = runTrace(trace: trace, input: Tensor<Float>(23.0))
    XCTAssertEqual(outputs.count, 1)
    XCTAssertEqual(outputs[0].valueDescription, "115.0")
  }

  func testNestedTracing() {
    func square(input: Tensor<Float>) -> Tensor<Float> {
      return input * input
    }

    func nestedTrace(input: Tensor<Float>) -> Tensor<Float> {
      let trace = LazyTensorTraceBuilder.trace(square)
      let outputs = runTrace(trace: trace, input: Tensor<Float>(3.0))
      XCTAssertEqual(outputs.count, 1)
      let handle = TensorHandle<Float>(handle: outputs[0])
      let y = Tensor<Float>(handle: handle)
      return y + input
    }

    let trace = LazyTensorTraceBuilder.trace(nestedTrace)
    XCTAssertEqual(
      trace.description,
      """
      lazyTrace_3(%0: float) -> (%2) {
        %1 = Const[dtype: float, value: 9.0]()
        %2 = AddV2[T: float](%1, %0)
      }
      """)
    let outputs = runTrace(trace: trace, input: Tensor<Float>(4.0))
    XCTAssertEqual(outputs.count, 1)
    XCTAssertEqual(outputs[0].valueDescription, "13.0")
  }

  func testCallableTrace() {
    func square(input: Tensor<Float>) -> Tensor<Float> {
      return input * input
    }
    let tracedSquare = _graph(square)
    XCTAssertEqual(tracedSquare(Tensor<Float>(10.0)).scalarized(), 100.0)
    XCTAssertEqual(tracedSquare(Tensor<Float>(5.0)).scalarized(), 25.0)
  }

  func testTraceWithOutputSameAsInput() {
    func identity(input: Tensor<Float>) -> Tensor<Float> { return input }
    let trace = LazyTensorTraceBuilder.trace(identity)
    XCTAssertEqual(
      trace.description,
      """
      lazyTrace_1(%0: float) -> (%0) {
      }
      """)
    let tracedIdentity = _graph(identity)
    XCTAssertEqual(tracedIdentity(Tensor<Float>(10.0)).scalarized(), 10.0)
    XCTAssertEqual(tracedIdentity(Tensor<Float>(17.0)).scalarized(), 17.0)
  }

  func testRetainsIdenticalOutputs() {
    typealias TensorFloatPair = Zip2TensorGroup<Tensor<Float>, Tensor<Float>>
    func makePair(input: Tensor<Float>) -> TensorFloatPair {
      return TensorFloatPair(input, input)
    }
    let trace = LazyTensorTraceBuilder.trace(makePair)
    XCTAssertEqual(
      trace.description,
      """
      lazyTrace_1(%0: float) -> (%0, %0) {
      }
      """)
    let tracedMakePair = _graph(makePair)
    let result = tracedMakePair(Tensor<Float>(5.0))
    XCTAssertEqual(result.first.scalarized(), 5.0)
    XCTAssertEqual(result.second.scalarized(), 5.0)
  }

  private func runTrace(trace: LazyTensorTrace, input: TensorGroup) -> [TFETensorHandle] {
    let tffunc = TFFunction(trace: trace)
    let inputHandles = input._tensorHandles.map { $0._tfeTensorHandle }
    let outputHandles = tffunc.execute(inputHandles)
    return outputHandles
  }

  static var allTests = [
    ("testSingleInput", testSingleInput),
    ("testTensorGroupInputOutputs", testTensorGroupInputOutputs),
    ("testClosureCapturesOfTensors", testClosureCapturesOfTensors),
    ("testClosureCapturesOfNonTensors", testClosureCapturesOfNonTensors),
    ("testNestedTracing", testNestedTracing),
    ("testCallableTrace", testCallableTrace),
    ("testTraceWithOutputSameAsInput", testTraceWithOutputSameAsInput),
    ("testRetainsIdenticalOutputs", testRetainsIdenticalOutputs),
  ]
}
