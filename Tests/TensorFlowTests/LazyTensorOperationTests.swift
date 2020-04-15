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

final class LazyTensorOperationTests: XCTestCase {

  func testNoInput() {
    let placeholder = LazyTensorOperation(
      _id: "V", name: "Placeholder", outputCount: 1)
    XCTAssertEqual(placeholder.description, "%V = Placeholder()")
    let placeholder2 = LazyTensorOperation(
      _id: "W", name: "Placeholder", outputCount: 2)
    XCTAssertEqual(placeholder2.description, "(%W.0, %W.1) = Placeholder()")
  }

  func testSingleInput() {
    let zero = Tensor<Float>(0.0)
    let zeroTFEHandle = zero.handle.handle._tfeTensorHandle

    let op0 = LazyTensorOperation(
      _id: "0", name: "Identity", outputCount: 1)
    op0.addInput(zeroTFEHandle)
    XCTAssertEqual(op0.description, "%0 = Identity(0.0)")

    let op1 = LazyTensorOperation(
      _id: "1", name: "Identity", outputCount: 1)
    op1.addInput(zero)
    XCTAssertEqual(op1.description, "%1 = Identity(0.0)")

    let op2 = LazyTensorOperation(
      _id: "2", name: "Identity", outputCount: 1)
    op2.addInput(StringTensor("hello"))
    XCTAssertEqual(op2.description, "%2 = Identity(\"string\")")

    let op3 = LazyTensorOperation(
      _id: "3", name: "Identity", outputCount: 1)
    let const = LazyTensorOperation(
      _id: "0", name: "Const", outputCount: 1)
    op3.addInput(LazyTensorHandle(_lazy: const, index: 0))
    XCTAssertEqual(op3.description, "%3 = Identity(%0)")
  }

  func testMultipleInputs() {
    let const = LazyTensorOperation(
      _id: "0", name: "Const", outputCount: 1)
    let op0 = LazyTensorOperation(
      _id: "0", name: "Identity", outputCount: 1)
    op0.addInput(LazyTensorHandle(_lazyLive: const, index: 0))
    op0.addInput(StringTensor("hello"))
    XCTAssertEqual(op0.description, "%0 = Identity(%0*, \"string\")")
  }

  func testListInputs() {
    let zero = Tensor<Float>(0.0)
    let tuple = LazyTensorOperation(
      _id: "0", name: "Tuple", outputCount: 3)
    let op0 = LazyTensorOperation(
      _id: "0", name: "IdentityN", outputCount: 2)
    let inputs: [TensorHandle<Float>] = [
      zero.handle,
      TensorHandle<Float>(handle: LazyTensorHandle(_lazy: tuple, index: 0)),
      TensorHandle<Float>(handle: LazyTensorHandle(_lazy: tuple, index: 1)),
    ]
    op0.addInputList(inputs)
    op0.addInput(LazyTensorHandle(_lazy: tuple, index: 2))
    XCTAssertEqual(
      op0.description, "(%0.0, %0.1) = IdentityN([0.0, %0.0, %0.1], %0.2)")

  }

  func testBoolAttribute() {
    let op0 = LazyTensorOperation(
      _id: "0", name: "Nop", outputCount: 1)
    op0.updateAttribute("b", true)
    XCTAssertEqual(op0.description, "%0 = Nop[b: true]()")
    op0.updateAttribute("b", false)
    XCTAssertEqual(op0.description, "%0 = Nop[b: false]()")
  }

  func testIntAttribute() {
    let op0 = LazyTensorOperation(
      _id: "0", name: "Nop", outputCount: 1)
    op0.updateAttribute("i", 10)
    XCTAssertEqual(op0.description, "%0 = Nop[i: Int(10)]()")
    op0.updateAttribute("i", 20)
    XCTAssertEqual(op0.description, "%0 = Nop[i: Int(20)]()")
  }

  func testFloatAttribute() {
    let op0 = LazyTensorOperation(
      _id: "0", name: "Nop", outputCount: 1)
    op0.updateAttribute("f", Float(10.0))
    XCTAssertEqual(op0.description, "%0 = Nop[f: Float(10.0)]()")
    op0.updateAttribute("f", Float(20.0))
    XCTAssertEqual(op0.description, "%0 = Nop[f: Float(20.0)]()")
  }

  func testDoubleAttribute() {
    let op0 = LazyTensorOperation(
      _id: "0", name: "Nop", outputCount: 1)
    op0.updateAttribute("d", Double(10.0))
    XCTAssertEqual(op0.description, "%0 = Nop[d: Double(10.0)]()")
    op0.updateAttribute("d", Double(20.0))
    XCTAssertEqual(op0.description, "%0 = Nop[d: Double(20.0)]()")
  }

  func testStringAttribute() {
    let op0 = LazyTensorOperation(
      _id: "0", name: "Nop", outputCount: 1)
    op0.updateAttribute("s", "hello")
    XCTAssertEqual(op0.description, "%0 = Nop[s: \"hello\"]()")
    op0.updateAttribute("s", "world")
    XCTAssertEqual(op0.description, "%0 = Nop[s: \"world\"]()")
  }

  func testTensorDataTypeAttribute() {
    let op0 = LazyTensorOperation(
      _id: "0", name: "Nop", outputCount: 1)
    op0.updateAttribute("a", TensorDataType(TF_INT32))
    XCTAssertEqual(op0.description, "%0 = Nop[a: int32]()")
    op0.updateAttribute("a", TensorDataType(TF_FLOAT))
    XCTAssertEqual(op0.description, "%0 = Nop[a: float]()")
    op0.updateAttribute("a", TensorDataType(TF_RESOURCE))
    XCTAssertEqual(op0.description, "%0 = Nop[a: resource]()")
  }

  func testTensorDataTypeArrayAttribute() {
    let op0 = LazyTensorOperation(
      _id: "0", name: "Nop", outputCount: 1)
    op0.updateAttribute("a", [TensorDataType(TF_INT32), TensorDataType(TF_FLOAT)])
    XCTAssertEqual(op0.description, "%0 = Nop[a: [int32, float]]()")
  }

  func testOptionalTensorShapeAttribute() {
    let op0 = LazyTensorOperation(
      _id: "0", name: "Nop", outputCount: 1)
    let shape1 = TensorShape([1, 2, 3])
    op0.updateAttribute("shape", shape1)
    XCTAssertEqual(op0.description, "%0 = Nop[shape: Optional([1, 2, 3])]()")
    op0.updateAttribute("shape", nil)
    XCTAssertEqual(op0.description, "%0 = Nop[shape: nil]()")
  }

  func testOptionalTensorShapeArrayAttribute() {
    let op0 = LazyTensorOperation(
      _id: "0", name: "Nop", outputCount: 1)
    let shape1 = TensorShape([1, 2, 3])
    let shape2 = TensorShape([4, 5])
    op0.updateAttribute("shapes", [shape1, shape2])
    XCTAssertEqual(
      op0.description,
      "%0 = Nop[shapes: [Optional([1, 2, 3]), Optional([4, 5])]]()")
    op0.updateAttribute("shapes", [nil, shape2])
    XCTAssertEqual(op0.description, "%0 = Nop[shapes: [nil, Optional([4, 5])]]()")
  }

  func testConstTensorAttribute() {
    let op0 = LazyTensorOperation(
      _id: "0", name: "Nop", outputCount: 1)
    let a = Tensor<Float>(5.5)
    let b = Tensor<Float>([1, 2])
    op0.updateAttribute("a", a.handle.handle._tfeTensorHandle)
    op0.updateAttribute("b", b.handle.handle._tfeTensorHandle)
    XCTAssertEqual(op0.description, "%0 = Nop[a: 5.5, b: [1.0, 2.0]]()")
  }

  func testArrayAttributes() {
    let op0 = LazyTensorOperation(
      _id: "0", name: "Nop", outputCount: 1)
    let bools: [Bool] = [true, false]
    op0.updateAttribute("a", bools)
    XCTAssertEqual(op0.description, "%0 = Nop[a: [true, false]]()")

    let ints: [Int] = [0, 1, 3]
    op0.updateAttribute("a", ints)
    XCTAssertEqual(op0.description, "%0 = Nop[a: Int[0, 1, 3]]()")

    let floats: [Float] = [0.0, 1.0, 2.0]
    op0.updateAttribute("a", floats)
    XCTAssertEqual(op0.description, "%0 = Nop[a: Float[0.0, 1.0, 2.0]]()")

    let doubles: [Double] = [0.0, 1.0, 4.0]
    op0.updateAttribute("a", doubles)
    XCTAssertEqual(op0.description, "%0 = Nop[a: Double[0.0, 1.0, 4.0]]()")

    let strings: [String] = ["a", "b", "c"]
    op0.updateAttribute("a", strings)
    XCTAssertEqual(op0.description, "%0 = Nop[a: String[a, b, c]]()")
  }

  func testMultipleAttributes() {
    let op0 = LazyTensorOperation(
      _id: "0", name: "Nop", outputCount: 1)
    op0.updateAttribute("b", true)
    op0.updateAttribute("s", "hello")
    XCTAssertEqual(op0.description, "%0 = Nop[b: true, s: \"hello\"]()")
  }

  func testFunctionAttribute() {
    let op0 = LazyTensorOperation(_id: "0", name: "Nop", outputCount: 1)
    op0.updateAttribute("fn", _TensorFunctionPointer(name: "ExampleFunction"))
    XCTAssertEqual(op0.description, "%0 = Nop[fn: TFFunction(ExampleFunction)]()")
  }

  func testDeviceTracking() {
    let op0 = LazyTensorOperation(_id: "0", name: "Nop", outputCount: 1)
    XCTAssertEqual(op0.deviceName, nil)
    withDevice(named: "/job:localhost/replica:0/task:0/device:CPU:0") {
      let op1 = LazyTensorOperation(_id: "0", name: "Nop", outputCount: 1)
      XCTAssertEqual(op1.deviceName ?? "", "/job:localhost/replica:0/task:0/device:CPU:0")
    }
  }

  static var allTests = [
    ("testNoInput", testNoInput),
    ("testSingleInput", testSingleInput),
    ("testMultipleInput", testMultipleInputs),
    ("testListInputs", testListInputs),
    ("testBoolAttribute", testBoolAttribute),
    ("testIntAttribute", testIntAttribute),
    ("testFloatAttribute", testFloatAttribute),
    ("testDoubleAttribute", testDoubleAttribute),
    ("testStringAttribute", testStringAttribute),
    ("testTensorDataTypeAttribute", testTensorDataTypeAttribute),
    ("testTensorDataTypeArrayAttribute", testTensorDataTypeArrayAttribute),
    (
      "testOptionalTensorShapeArrayAttribute",
      testOptionalTensorShapeArrayAttribute
    ),
    ("testOptionalTensorShapeAttribute", testOptionalTensorShapeAttribute),
    (
      "testTensorShapeArrayAttribute",
      testOptionalTensorShapeArrayAttribute
    ),
    ("testConstTensorAttribute", testConstTensorAttribute),
    ("testArrayAttributes", testArrayAttributes),
    ("testMultipleAttributes", testMultipleAttributes),
    ("testFunctionAttribute", testFunctionAttribute),
    ("testDeviceTracking", testDeviceTracking),

  ]
}
