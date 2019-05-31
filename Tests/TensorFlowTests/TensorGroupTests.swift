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

extension TensorDataType : Equatable {
  public static func == (lhs: TensorDataType, rhs: TensorDataType) -> Bool {
    return Int(lhs._cDataType.rawValue) == Int(rhs._cDataType.rawValue)
  }
}

struct Empty : TensorGroup {}

struct Simple : TensorGroup, Equatable {
  var w, b: Tensor<Float>
}

struct Mixed : TensorGroup, Equatable {
  // Mutable.
  var float: Tensor<Float>
  // Immutable.
  let int: Tensor<Int32>
}

struct Nested : TensorGroup, Equatable {
  // Immutable.
  let simple: Simple
  // Mutable.
  var mixed: Mixed
}

struct Generic<T: TensorGroup & Equatable, U: TensorGroup & Equatable> : TensorGroup, Equatable {
  var t: T
  var u: U
}

final class TensorGroupTests: XCTestCase {
    func testEmptyList() {
        XCTAssertEqual([], Empty._typeList)
    }

    func testSimpleTypeList() {
        let float = Float.tensorFlowDataType
        XCTAssertEqual([float, float], Simple._typeList)
    }

    func testSimpleInit() {
        let w = Tensor<Float>(0.1)
        let b = Tensor<Float>(0.1)
        let simple = Simple(w: w, b: b)
        
        let status = TF_NewStatus()
        let wHandle = TFE_TensorHandleCopySharingTensor(
            w.handle._cTensorHandle, status)!
        let bHandle = TFE_TensorHandleCopySharingTensor(
            b.handle._cTensorHandle, status)!
        TF_DeleteStatus(status)
        
        let buffer = UnsafeMutableBufferPointer<CTensorHandle>.allocate(
            capacity: 2)
        let _ = buffer.initialize(from: [wHandle, bHandle])
        let expectedSimple = Simple(_owning: UnsafePointer(buffer.baseAddress))
        
        XCTAssertEqual(expectedSimple, simple)
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
        
        let status = TF_NewStatus()
        let floatHandle = TFE_TensorHandleCopySharingTensor(
            float.handle._cTensorHandle, status)!
        let intHandle = TFE_TensorHandleCopySharingTensor(
            int.handle._cTensorHandle, status)!
        TF_DeleteStatus(status)
        
        let buffer = UnsafeMutableBufferPointer<CTensorHandle>.allocate(
            capacity: 2)
        let _ = buffer.initialize(from: [floatHandle, intHandle])
        let expectedMixed = Mixed(_owning: UnsafePointer(buffer.baseAddress))
        
        XCTAssertEqual(expectedMixed, mixed)
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
        
        let status = TF_NewStatus()
        let wHandle = TFE_TensorHandleCopySharingTensor(
            w.handle._cTensorHandle, status)!
        let bHandle = TFE_TensorHandleCopySharingTensor(
            b.handle._cTensorHandle, status)!
        let floatHandle = TFE_TensorHandleCopySharingTensor(
            float.handle._cTensorHandle, status)!
        let intHandle = TFE_TensorHandleCopySharingTensor(
            int.handle._cTensorHandle, status)!
        TF_DeleteStatus(status)
        
        let buffer = UnsafeMutableBufferPointer<CTensorHandle>.allocate(
            capacity: 4)
        let _ = buffer.initialize(
            from: [wHandle, bHandle, floatHandle, intHandle])
        let expectedNested = Nested(
            _owning: UnsafePointer(buffer.baseAddress))
        
        XCTAssertEqual(expectedNested, nested)
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
        
        let status = TF_NewStatus()
        let wHandle = TFE_TensorHandleCopySharingTensor(
            w.handle._cTensorHandle, status)!
        let bHandle = TFE_TensorHandleCopySharingTensor(
            b.handle._cTensorHandle, status)!
        let floatHandle = TFE_TensorHandleCopySharingTensor(
            float.handle._cTensorHandle, status)!
        let intHandle = TFE_TensorHandleCopySharingTensor(
            int.handle._cTensorHandle, status)!
        TF_DeleteStatus(status)
        
        let buffer = UnsafeMutableBufferPointer<CTensorHandle>.allocate(
            capacity: 4)
        let _ = buffer.initialize(
            from: [wHandle, bHandle, floatHandle, intHandle])
        let expectedGeneric = Generic<Simple, Mixed>(
            _owning: UnsafePointer(buffer.baseAddress))
        
        XCTAssertEqual(expectedGeneric, generic)
    }
    
    func testNestedGenericTypeList() {
        struct NestedGeneric {
            func function() {
                struct UltraNested<
                    T: TensorGroup & Equatable, V: TensorGroup & Equatable>
                : TensorGroup, Equatable {
                    var a: Generic<T, V>
                    var b: Generic<V, T>
                }
                let float = Float.tensorFlowDataType
                let int = Int32.tensorFlowDataType
                XCTAssertEqual([float, float, float, int, float, int, float, float],
                    UltraNested<Simple, Mixed>._typeList)
            }
        }
        
        NestedGeneric().function()
    }
    
    func testNestedGenericInit() {
        struct NestedGeneric {
            func function() {
                struct UltraNested<
                    T: TensorGroup & Equatable, V: TensorGroup & Equatable>
                : TensorGroup, Equatable {
                    var a: Generic<T, V>
                    var b: Generic<V, T>
                }
                
                let w = Tensor<Float>(0.1)
                let b = Tensor<Float>(0.1)
                let simple = Simple(w: w, b: b)
                let float = Tensor<Float>(0.1)
                let int = Tensor<Int32>(1)
                let mixed = Mixed(float: float, int: int)
                let genericSM = Generic<Simple, Mixed>(t: simple, u: mixed)
                let genericMS = Generic<Mixed, Simple>(t: mixed, u: simple)
                let generic = UltraNested(a: genericSM, b: genericMS)
                
                let status = TF_NewStatus()
                let wHandle1 = TFE_TensorHandleCopySharingTensor(w.handle._cTensorHandle, status)!
                let wHandle2 = TFE_TensorHandleCopySharingTensor(w.handle._cTensorHandle, status)!
                let bHandle1 = TFE_TensorHandleCopySharingTensor(b.handle._cTensorHandle, status)!
                let bHandle2 = TFE_TensorHandleCopySharingTensor(b.handle._cTensorHandle, status)!
                let floatHandle1 = TFE_TensorHandleCopySharingTensor(float.handle._cTensorHandle, status)!
                let floatHandle2 = TFE_TensorHandleCopySharingTensor(float.handle._cTensorHandle, status)!
                let intHandle1 = TFE_TensorHandleCopySharingTensor(int.handle._cTensorHandle, status)!
                let intHandle2 = TFE_TensorHandleCopySharingTensor(int.handle._cTensorHandle, status)!
                TF_DeleteStatus(status)
                
                let buffer = UnsafeMutableBufferPointer<CTensorHandle>.allocate(capacity: 8)
                let _ = buffer.initialize(from: [wHandle1, bHandle1, floatHandle1,  intHandle1,
                        floatHandle2, intHandle2, wHandle2, bHandle2])
                let expectedGeneric = UltraNested<Simple, Mixed>(
                    _owning: UnsafePointer(buffer.baseAddress))
                
                XCTAssertEqual(expectedGeneric, generic)
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
        ("testNestedGenericInit", testNestedGenericInit)
    ]

}
