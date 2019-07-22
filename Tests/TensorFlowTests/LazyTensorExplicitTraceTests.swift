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

final class LazyTensorExplicitTraceTests: XCTestCase {
    override class func setUp() {
        super.setUp()
        _RuntimeConfig.useLazyTensor = true
    }

    override class func tearDown() {
        super.tearDown()
        _RuntimeConfig.useLazyTensor = false
    }

    func testSingleInput() {
        func fn(x: Tensor<Float>) -> Tensor<Float> { return x + x }
        let trace = LazyTensorTraceBuilder.trace(fn)
        XCTAssertEqual(trace.description,
            """
            lazyTrace_2(%0: float) -> (%1) {
              %1 = Add[T: float](%0, %0)
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
        XCTAssertEqual(trace.description,
            """
            lazyTrace_6(%0: float, %1: int32) -> (%3, %5) {
              %2 = Const[dtype: int32, value: 4]()
              %3 = Mul[T: int32](%1, %2)
              %4 = Const[dtype: float, value: 3.0]()
              %5 = Add[T: float](%0, %4)
            }
            """)
        let outputs = runTrace(
            trace: trace,
            input: TensorFloatInt32Pair(Tensor<Float>(10.0), Tensor<Int32>(5)))
        XCTAssertEqual(outputs.count, 2)
        XCTAssertEqual(outputs[0].valueDescription, "20")
        XCTAssertEqual(outputs[1].valueDescription, "13.0")
    }

    private func runTrace(trace: LazyTensorTrace, input: TensorGroup) -> [TFETensorHandle] {
        let tffunc = TFFunction(trace: trace)
        let inputHandles = input._tensorHandles.map { $0._tfeTensorHandle }
        let outputHandles = tffunc.execute(inputHandles)
        return outputHandles
    }

    static var allTests = [
        ("testSingleInput", testSingleInput),
        ("testTensorGroupInputOutputs", testTensorGroupInputOutputs)
    ]
}
