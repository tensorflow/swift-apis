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

struct Empty: TensorGroup {}

struct Simple: TensorGroup, Equatable {
  var w, b: Tensor<Float>
}

struct Mixed: TensorGroup, Equatable {
  // Mutable.
  var float: Tensor<Float>
  // Immutable.
  let int: Tensor<Int32>
}

struct Nested: TensorGroup, Equatable {
  // Immutable.
  let simple: Simple
  // Mutable.
  var mixed: Mixed
}

struct Generic<T: TensorGroup & Equatable, U: TensorGroup & Equatable>: TensorGroup, Equatable {
  var t: T
  var u: U
}

struct UltraNested<T: TensorGroup & Equatable, V: TensorGroup & Equatable>: TensorGroup, Equatable {
  var a: Generic<T, V>
  var b: Generic<V, T>
}

extension TensorHandle {
  func makeCopy() -> TFETensorHandle {
    let status = TF_NewStatus()
    let result = TFETensorHandle(
      _owning: TFE_TensorHandleCopySharingTensor(handle._cTensorHandle, status)!)
    XCTAssertEqual(TF_GetCode(status), TF_OK)
    TF_DeleteStatus(status)
    return result
  }
}

extension TensorArrayProtocol {
  var tfeTensorHandles: [TFETensorHandle] {
    self._tensorHandles.map { $0 as! TFETensorHandle }
  }
}

final class TensorGroupTests: XCTestCase {
  func testEmptyList() {
    XCTAssertEqual([], Empty._typeList)
    XCTAssertEqual(Empty()._tensorHandles.count, 0)
  }

  func testSimpleTypeList() {
    let float = Float.tensorFlowDataType
    XCTAssertEqual([float, float], Simple._typeList)
  }

  func testSimpleInit() {
    let w = Tensor<Float>(0.1)
    let b = Tensor<Float>(0.1)
    let simple = Simple(w: w, b: b)

    let wHandle = w.handle.makeCopy()
    let bHandle = b.handle.makeCopy()

    let expectedSimple = Simple(_handles: [wHandle, bHandle])
    XCTAssertEqual(expectedSimple, simple)

    let reconstructedSimple = Simple(_handles: simple.tfeTensorHandles)
    XCTAssertEqual(reconstructedSimple, simple)
  }

  func testMixedTypeList() {
    let float = Float.tensorFlowDataType
    let int = Int32.tensorFlowDataType
    XCTAssertEqual([float, int], Mixed._typeList)
  }

  func testMixedInit() {
    let float = Tensor<Float>(0.1)
    let int = Tensor<Int32>(1)
    let mixed = Mixed(float: float, int: int)

    let floatHandle = float.handle.makeCopy()
    let intHandle = int.handle.makeCopy()

    let expectedMixed = Mixed(_handles: [floatHandle, intHandle])
    XCTAssertEqual(expectedMixed, mixed)

    let reconstructedMixed = Mixed(_handles: mixed.tfeTensorHandles)
    XCTAssertEqual(reconstructedMixed, mixed)
  }

  func testNestedTypeList() {
    let float = Float.tensorFlowDataType
    let int = Int32.tensorFlowDataType
    XCTAssertEqual([float, float, float, int], Nested._typeList)
  }

  func testNestedInit() {
    let w = Tensor<Float>(0.1)
    let b = Tensor<Float>(0.1)
    let simple = Simple(w: w, b: b)
    let float = Tensor<Float>(0.1)
    let int = Tensor<Int32>(1)
    let mixed = Mixed(float: float, int: int)
    let nested = Nested(simple: simple, mixed: mixed)

    let wHandle = w.handle.makeCopy()
    let bHandle = b.handle.makeCopy()
    let floatHandle = float.handle.makeCopy()
    let intHandle = int.handle.makeCopy()

    let expectedNested = Nested(
      _handles: [wHandle, bHandle, floatHandle, intHandle])
    XCTAssertEqual(expectedNested, nested)

    let reconstructedNested = Nested(_handles: nested.tfeTensorHandles)
    XCTAssertEqual(reconstructedNested, nested)
  }

  func testGenericTypeList() {
    let float = Float.tensorFlowDataType
    let int = Int32.tensorFlowDataType
    XCTAssertEqual(
      [float, float, float, int], Generic<Simple, Mixed>._typeList)
  }

  func testGenericInit() {
    let w = Tensor<Float>(0.1)
    let b = Tensor<Float>(0.1)
    let simple = Simple(w: w, b: b)
    let float = Tensor<Float>(0.1)
    let int = Tensor<Int32>(1)
    let mixed = Mixed(float: float, int: int)
    let generic = Generic(t: simple, u: mixed)

    let wHandle = w.handle.makeCopy()
    let bHandle = b.handle.makeCopy()
    let floatHandle = float.handle.makeCopy()
    let intHandle = int.handle.makeCopy()

    let expectedGeneric = Generic<Simple, Mixed>(
      _handles: [wHandle, bHandle, floatHandle, intHandle])
    XCTAssertEqual(expectedGeneric, generic)

    let reconstructedGeneric = Generic<Simple, Mixed>(_handles: generic.tfeTensorHandles)
    XCTAssertEqual(reconstructedGeneric, generic)
  }

  func testNestedGenericTypeList() {
    struct NestedGeneric {
      func function() {
        let float = Float.tensorFlowDataType
        let int = Int32.tensorFlowDataType
        XCTAssertEqual(
          [float, float, float, int, float, int, float, float],
          UltraNested<Simple, Mixed>._typeList)
      }
    }

    NestedGeneric().function()
  }

  func testNestedGenericInit() {
    struct NestedGeneric {
      func function() {
        let w = Tensor<Float>(0.1)
        let b = Tensor<Float>(0.1)
        let simple = Simple(w: w, b: b)
        let float = Tensor<Float>(0.1)
        let int = Tensor<Int32>(1)
        let mixed = Mixed(float: float, int: int)
        let genericSM = Generic<Simple, Mixed>(t: simple, u: mixed)
        let genericMS = Generic<Mixed, Simple>(t: mixed, u: simple)
        let generic = UltraNested(a: genericSM, b: genericMS)

        let wHandle1 = w.handle.makeCopy()
        let wHandle2 = w.handle.makeCopy()
        let bHandle1 = b.handle.makeCopy()
        let bHandle2 = b.handle.makeCopy()
        let floatHandle1 = float.handle.makeCopy()
        let floatHandle2 = float.handle.makeCopy()
        let intHandle1 = int.handle.makeCopy()
        let intHandle2 = int.handle.makeCopy()

        let expectedGeneric = UltraNested<Simple, Mixed>(
          _handles: [
            wHandle1, bHandle1, floatHandle1, intHandle1,
            floatHandle2, intHandle2, wHandle2, bHandle2,
          ])
        XCTAssertEqual(expectedGeneric, generic)

        let reconstructedGeneric = UltraNested<Simple, Mixed>(
          _handles: generic.tfeTensorHandles)
        XCTAssertEqual(reconstructedGeneric, generic)
      }
    }

    NestedGeneric().function()
  }

  static var allTests = [
    ("testEmptyList", testEmptyList),
    ("testSimpleTypeList", testSimpleTypeList),
    ("testSimpleInit", testSimpleInit),
    ("testMixedTypelist", testMixedTypeList),
    ("testMixedInit", testMixedInit),
    ("testNestedTypeList", testNestedTypeList),
    ("testNestedInit", testNestedInit),
    ("testGenericTypeList", testGenericTypeList),
    ("testGenericInit", testGenericInit),
    ("testNestedGenericTypeList", testNestedGenericTypeList),
    ("testNestedGenericInit", testNestedGenericInit),
  ]

}
