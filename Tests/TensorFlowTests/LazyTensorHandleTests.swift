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

struct LazyTensorOperationRef: Equatable, Hashable {
  let value: LazyTensorOperation
  init(_ value: LazyTensorOperation) { self.value = value }
  func hash(into hasher: inout Hasher) {
    hasher.combine(ObjectIdentifier(value))
  }
}

func == (lhs: LazyTensorOperationRef, rhs: LazyTensorOperationRef) -> Bool {
  return lhs.value === rhs.value
}

final class LazyTensorHandleTests: XCTestCase {
  func testConstructions() {
    let zero = Tensor<Float>(0.0)
    let zeroTFEHandle = zero.handle.handle._tfeTensorHandle

    let concTensor = LazyTensorHandle(zeroTFEHandle)
    XCTAssertEqual(concTensor.description, "0.0")

    let materializedConcTensor = LazyTensorHandle(
      _materialized: zeroTFEHandle)
    XCTAssertEqual(materializedConcTensor.description, "0.0*")

    let op = LazyTensorOperation(
      _id: "0", name: "IdentityN", outputCount: 3)
    let symTensor0 = LazyTensorHandle(_lazy: op, index: 0)
    XCTAssertEqual(symTensor0.description, "%0.0")

    let symTensor1 = LazyTensorHandle(_lazy: op, index: 2)
    XCTAssertEqual(symTensor1.description, "%0.2")

    let liveSymTensor = LazyTensorHandle(_lazyLive: op, index: 0)
    XCTAssertEqual(liveSymTensor.description, "%0.0*")
  }

  func testLazyTensorOperationProperty() {
    let zero = Tensor<Float>(0.0)
    let zeroTFEHandle = zero.handle.handle._tfeTensorHandle
    let concTensor = LazyTensorHandle(zeroTFEHandle)
    XCTAssertNil(concTensor.lazyTensorOperation)

    let op = LazyTensorOperation(
      _id: "0", name: "IdentityN", outputCount: 3)
    let symTensor = LazyTensorHandle(_lazy: op, index: 0)
    let lazyTensorOperation = symTensor.lazyTensorOperation
    XCTAssertNotNil(lazyTensorOperation)
    // Checks that returned value is the same as the one that we passed in.
    XCTAssertTrue(lazyTensorOperation === op)
  }

  func testLivenessTracking() {
    func assertLive(_ expectedLive: [LazyTensorOperation]) {
      var actualLiveOps: Set<LazyTensorOperationRef> = []
      LazyTensorHandle.forEachLiveOperation {
        actualLiveOps.insert(LazyTensorOperationRef($0))
      }
      let expectedLiveOps = Set<LazyTensorOperationRef>(
        expectedLive.map { LazyTensorOperationRef($0) }
      )
      XCTAssertEqual(actualLiveOps, expectedLiveOps)
    }

    func assertAll(_ expectedAll: [LazyTensorOperation]) {
      var actualAllOps: Set<LazyTensorOperationRef> = []
      LazyTensorHandle.forEachOperation {
        actualAllOps.insert(LazyTensorOperationRef($0))
      }
      let expectedAllOps = Set<LazyTensorOperationRef>(
        expectedAll.map { LazyTensorOperationRef($0) }
      )
      XCTAssertEqual(actualAllOps, expectedAllOps)
    }

    let op0 = LazyTensorOperation(
      _id: "0", name: "IdentityN", outputCount: 2)
    let op1 = LazyTensorOperation(
      _id: "1", name: "IdentityN", outputCount: 2)

    XCTAssertFalse(LazyTensorHandle.isLive(op0))
    XCTAssertFalse(LazyTensorHandle.isLive(op1))

    let t0 = LazyTensorHandle(_lazyLive: op0, index: 0)
    let t1 = LazyTensorHandle(_lazy: op1, index: 1)
    XCTAssertTrue(LazyTensorHandle.isLive(op0))
    XCTAssertFalse(LazyTensorHandle.isLive(op1))

    do {
      let t3 = LazyTensorHandle(_lazyLive: op1, index: 0)
      XCTAssertTrue(LazyTensorHandle.isLive(op1))
      assertLive([op0, op1])
      assertAll([op0, op1])
      // The following is here just to ensure t3 is live.
      XCTAssertTrue(isSymbolic(t3))
    }
    XCTAssertFalse(LazyTensorHandle.isLive(op1))
    assertLive([op0])
    assertAll([op0, op1])

    // The following are here just to ensure t0 and t1 are live.
    XCTAssertTrue(isSymbolic(t1))
    XCTAssertTrue(isSymbolic(t0))
  }

  private func checkConversions<T: _LazyTensorCompatible>(_ x: T) {
    let concreteLazyX = x._concreteLazyTensor
    let concreteInputLazyX = x._concreteInputLazyTensor
    XCTAssertFalse(isSymbolic(concreteLazyX._lazyTensorHandle))
    XCTAssertFalse(isSymbolic(concreteInputLazyX._lazyTensorHandle))
    XCTAssertFalse(isMaterializedConcrete(concreteLazyX._lazyTensorHandle))
    XCTAssertTrue(isMaterializedConcrete(concreteInputLazyX._lazyTensorHandle))
  }

  func testTensorToLazyTensorConversions() {
    checkConversions(Tensor<Float>(10.0))
    checkConversions(StringTensor("Hello!"))

    // ResourceHandle and VariantHandle conversions.
    let elements1: Tensor<Int32> = [0, 1, 2]
    let elements2: Tensor<Int32> = [10, 11, 12]
    let outputTypes = [Int32.tensorFlowDataType, Int32.tensorFlowDataType]
    let outputShapes: [TensorShape?] = [nil, nil]
    let dataset: VariantHandle = _Raw.tensorSliceDataset(
      components: [elements1, elements2],
      outputShapes: outputShapes
    )
    let iterator: ResourceHandle = _Raw.iteratorV2(
      sharedName: "blah",
      container: "earth", outputTypes: outputTypes, outputShapes: outputShapes
    )
    checkConversions(dataset)
    checkConversions(iterator)
  }

  private func isSymbolic(_ t: LazyTensorHandle?) -> Bool {
    guard let t = t else { return false }
    switch t.handle {
    case .symbolic: return true
    case .concrete: return false
    }
  }

  private func isMaterializedConcrete(_ t: LazyTensorHandle?) -> Bool {
    guard let t = t else { return false }
    switch t.handle {
    case .symbolic: return true
    case .concrete(_, let isMaterialized): return isMaterialized
    }
  }

  static var allTests = [
    ("testConstructions", testConstructions),
    ("testLazyTensorOperationProperty", testLazyTensorOperationProperty),
    ("testLivenessTracking", testLivenessTracking),
    ("testTensorToLazyTensorConversions", testTensorToLazyTensorConversions),
  ]
}
