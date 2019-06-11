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

final class LazyTensorOperationTests: XCTestCase {
    func testNoInput() {
        let placeholder = LazyTensorOperation(
            _id: "V", name: "Placeholder", outputCount: 1)
        XCTAssertEqual("\(placeholder)", "Placeholder_V[]():1")
        let placeholder2 = LazyTensorOperation(
            _id: "W", name: "Placeholder", outputCount: 2)
        XCTAssertEqual("\(placeholder2)", "Placeholder_W[]():2")
    }

    func testSingleInput() {
        let zero = Tensor<Float>(0.0)
        let zeroTFEHandle = zero.handle.handle._tfeTensorHandle

        let op0 = LazyTensorOperation(
            _id: "0", name: "Identity", outputCount: 1)
        op0.addInput(zeroTFEHandle)
        XCTAssertEqual(op0.description, "Identity_0[](0.0):1")

        let op1 = LazyTensorOperation(
            _id: "1", name: "Identity", outputCount: 1)
        op1.addInput(zero)
        XCTAssertEqual(op1.description, "Identity_1[](0.0):1")

        let op2 = LazyTensorOperation(
            _id: "2", name: "Identity", outputCount: 1)
        op2.addInput(StringTensor("hello"))
        XCTAssertEqual(op2.description, "Identity_2[](\"string\"):1")

        let op3 = LazyTensorOperation(
            _id: "3", name: "Identity", outputCount: 1)
        let const = LazyTensorOperation(
            _id: "0", name: "Const", outputCount: 1)
        op3.addInput(LazyTensor(_lazy: const, index: 0))
        XCTAssertEqual(op3.description, "Identity_3[](Const_0:0):1")
    }

    func testMultipleInputs() {
        let const = LazyTensorOperation(
            _id: "0", name: "Const", outputCount: 1)
        let op0 = LazyTensorOperation(
            _id: "0", name: "Identity", outputCount: 1)
        op0.addInput(LazyTensor(_lazyLive: const, index: 0))
        op0.addInput(StringTensor("hello"))
        XCTAssertEqual(op0.description, "Identity_0[](Const_0:0*, \"string\"):1")
    }


    func testListInputs() {
        let zero = Tensor<Float>(0.0)
        let tuple = LazyTensorOperation(
            _id: "0", name: "Tuple", outputCount: 3)
        let op0 = LazyTensorOperation(
            _id: "0", name: "IdentityN", outputCount: 2)
        let inputs: [TensorHandle<Float>] = [
            zero.handle,
            TensorHandle<Float>(handle: LazyTensor(_lazy: tuple, index: 0)),
            TensorHandle<Float>(handle: LazyTensor(_lazy: tuple, index: 1))]
        op0.addInputList(inputs)
        op0.addInput(LazyTensor(_lazy: tuple, index: 2))
        XCTAssertEqual(
            op0.description,
            "Identity_0[]([conc, Tuple_0:0, Tuple_0:1*], Tuple_0:2):2")

    }

    func testBoolAttribute() {
        let op0 = LazyTensorOperation(
            _id: "0", name: "Nop", outputCount: 2)
        op0.updateAttribute("b", true)
        XCTAssertEqual(op0.description, "Nop_0[b: true]():2")
        op0.updateAttribute("b", false)
        XCTAssertEqual(op0.description, "Nop_0[b: false]():2")
    }

    func testIntAttribute() {
        let op0 = LazyTensorOperation(
            _id: "0", name: "Nop", outputCount: 2)
        op0.updateAttribute("i", 10)
        XCTAssertEqual(op0.description, "Nop_0[i: Int(10)]():2")
        op0.updateAttribute("i", 20)
        XCTAssertEqual(op0.description, "Nop_0[i: Int(20)]():2")
    }

    func testFloatAttribute() {
        let op0 = LazyTensorOperation(
            _id: "0", name: "Nop", outputCount: 2)
        op0.updateAttribute("f", Float(10.0))
        XCTAssertEqual(op0.description, "Nop_0[f: Float(10.0)]():2")
        op0.updateAttribute("f", Float(20.0))
        XCTAssertEqual(op0.description, "Nop_0[f: Float(20.0)]():2")
    }

    func testDoubleAttribute() {
        let op0 = LazyTensorOperation(
            _id: "0", name: "Nop", outputCount: 2)
        op0.updateAttribute("d", Double(10.0))
        XCTAssertEqual(op0.description, "Nop_0[d: Double(10.0)]():2")
        op0.updateAttribute("d", Double(20.0))
        XCTAssertEqual(op0.description, "Nop_0[d: Double(20.0)]():2")
    }

    func testStringAttribute() {
        let op0 = LazyTensorOperation(
            _id: "0", name: "Nop", outputCount: 2)
        op0.updateAttribute("s", "hello")
        XCTAssertEqual(op0.description, "Nop_0[s: \"hello\"]():2")
        op0.updateAttribute("s", "world")
        XCTAssertEqual(op0.description, "Nop_0[s: \"world\"]():2")
    }

    func testTensorDataTypeAttribute() {
        let op0 = LazyTensorOperation(
            _id: "0", name: "Nop", outputCount: 2)
        op0.updateAttribute("a", TensorDataType(TF_INT32))
        XCTAssertEqual(op0.description, "Nop_0[a: int32]():2")
        op0.updateAttribute("a", TensorDataType(TF_FLOAT))
        XCTAssertEqual(op0.description, "Nop_0[a: float]():2")
        op0.updateAttribute("a", TensorDataType(TF_RESOURCE))
        XCTAssertEqual(op0.description, "Nop_0[a: resource]():2")
    }

    func testArrayAttributes() {
        let op0 = LazyTensorOperation(
            _id: "0", name: "Nop", outputCount: 2)
        let bools: [Bool] = [true, false]
        op0.updateAttribute("a", bools)
        XCTAssertEqual(op0.description, "Nop_0[a: [true, false]]():2")

        let ints: [Int] = [0, 1, 3]
        op0.updateAttribute("a", ints)
        XCTAssertEqual(op0.description, "Nop_0[a: Int[0, 1, 3]]():2")

        let floats: [Float] = [0.0, 1.0, 2.0]
        op0.updateAttribute("a", floats)
        XCTAssertEqual(op0.description, "Nop_0[a: Float[0.0, 1.0, 2.0]]():2")

        let doubles: [Double] = [0.0, 1.0, 4.0]
        op0.updateAttribute("a", doubles)
        XCTAssertEqual(op0.description, "Nop_0[a: Double[0.0, 1.0, 4.0]]():2")

        let strings: [String] = ["a", "b", "c"]
        op0.updateAttribute("a", strings)
        XCTAssertEqual(op0.description, "Nop_0[a: String[a, b, c]]():2")
    }

    func testMultipleAttributes() {
        let op0 = LazyTensorOperation(
            _id: "0", name: "Nop", outputCount: 2)
        op0.updateAttribute("b", true)
        op0.updateAttribute("s", "hello")
        XCTAssert(
            (op0.description == "Nop_0[b: true, s: \"hello\"]():2") ||
            (op0.description == "Nop_0[s: \"hello\", b: true]():2")
        )
    }

    static var allTests = [
        ("testNoInput", testNoInput),
        ("testSingleInput", testSingleInput),
        ("testMultipleInput", testMultipleInputs),
        ("testBoolAttribute", testBoolAttribute),
        ("testIntAttribute", testIntAttribute),
        ("testFloatAttribute", testFloatAttribute),
        ("testDoubleAttribute", testDoubleAttribute),
        ("testStringAttribute", testStringAttribute),
        ("testTensorDataTypeAttribute", testTensorDataTypeAttribute),
        ("testArrayAttributes", testArrayAttributes),
        ("testMultipleAttributes", testMultipleAttributes)
    ]
}
