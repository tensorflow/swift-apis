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

struct Empty : TensorGroup {
    init() {}
    public init<C: RandomAccessCollection>(
        _handles: C
    ) where C.Element: _AnyTensorHandle {}
    public var _tensorHandles: [_AnyTensorHandle] { [] }
}

struct Simple : TensorGroup, Equatable {
    var w, b: Tensor<Float>

    init(w: Tensor<Float>, b: Tensor<Float>) {
        self.w = w
        self.b = b
    }

    public init<C: RandomAccessCollection>(
        _handles: C
    ) where C.Element: _AnyTensorHandle {
        precondition(_handles.count == 2)
        let wIndex = _handles.startIndex
        let bIndex = _handles.index(wIndex, offsetBy: 1)
        w = Tensor<Float>(handle: TensorHandle<Float>(handle: _handles[wIndex]))
        b = Tensor<Float>(handle: TensorHandle<Float>(handle: _handles[bIndex]))
    }

    public var _tensorHandles: [_AnyTensorHandle] { [w.handle.handle, b.handle.handle] }
}

struct Mixed : TensorGroup, Equatable {
    // Mutable.
    var float: Tensor<Float>
    // Immutable.
    let int: Tensor<Int32>

    init(float: Tensor<Float>, int: Tensor<Int32>) {
        self.float = float
        self.int = int
    }

    public init<C: RandomAccessCollection>(
        _handles: C
    ) where C.Element: _AnyTensorHandle {
        precondition(_handles.count == 2)
        let floatIndex = _handles.startIndex
        let intIndex = _handles.index(floatIndex, offsetBy: 1)
        float = Tensor<Float>(
            handle: TensorHandle<Float>(handle: _handles[floatIndex]))
        int = Tensor<Int32>(
            handle: TensorHandle<Int32>(handle: _handles[intIndex]))
    }

    public var _tensorHandles: [_AnyTensorHandle] {
        [float.handle.handle, int.handle.handle]
    }
}

struct Nested : TensorGroup, Equatable {
    // Immutable.
    let simple: Simple
    // Mutable.
    var mixed: Mixed

    init(simple: Simple, mixed: Mixed) {
        self.simple = simple
        self.mixed = mixed
    }

    public init<C: RandomAccessCollection>(
        _handles: C
    ) where C.Element: _AnyTensorHandle {
        let simpleStart = _handles.startIndex
        let simpleEnd = _handles.index(
            simpleStart, offsetBy: Int(Simple._tensorHandleCount))
        simple = Simple(_handles: _handles[simpleStart..<simpleEnd])
        mixed = Mixed(_handles: _handles[simpleEnd..<_handles.endIndex])
    }

    public var _tensorHandles: [_AnyTensorHandle] {
        simple._tensorHandles + mixed._tensorHandles
    }
}

struct Generic<T: TensorGroup & Equatable, U: TensorGroup & Equatable> : TensorGroup, Equatable {
    var t: T
    var u: U

    public init(t: T, u: U) {
        self.t = t
        self.u = u
    }

    public init<C: RandomAccessCollection>(
        _handles: C
    ) where C.Element: _AnyTensorHandle {
        let tStart = _handles.startIndex
        let tEnd = _handles.index(tStart, offsetBy: Int(T._tensorHandleCount))
        t = T.init(_handles: _handles[tStart..<tEnd])
        u = U.init(_handles: _handles[tEnd..<_handles.endIndex])
    }

    public var _tensorHandles: [_AnyTensorHandle] {
        t._tensorHandles + u._tensorHandles
    }
}

struct UltraNested<T: TensorGroup & Equatable, V: TensorGroup & Equatable>
    : TensorGroup, Equatable {
    var a: Generic<T, V>
    var b: Generic<V, T>

    init(a: Generic<T, V>, b: Generic<V,T>) {
        self.a = a
        self.b = b
    }

    public init<C: RandomAccessCollection>(
        _handles: C
    ) where C.Element: _AnyTensorHandle {
        let firstStart = _handles.startIndex
        let firstEnd = _handles.index(
            firstStart, offsetBy: Int(Generic<T,V>._tensorHandleCount))
        a = Generic<T,V>.init(_handles: _handles[firstStart..<firstEnd])
        b = Generic<V,T>.init(_handles: _handles[firstEnd..<_handles.endIndex])
    }

    public var _tensorHandles: [_AnyTensorHandle] {
        return a._tensorHandles + b._tensorHandles
    }
}

func copy<T>(of handle: TensorHandle<T>) -> TFETensorHandle {
    let status = TF_NewStatus()
    let result = TFETensorHandle(_owning: TFE_TensorHandleCopySharingTensor(
            handle._cTensorHandle, status)!)
    XCTAssertEqual(TF_GetCode(status), TF_OK)
    TF_DeleteStatus(status)
    return result
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

        let wHandle = copy(of: w.handle)
        let bHandle = copy(of: b.handle)

        let expectedSimple = Simple(_handles: [wHandle, bHandle])

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

        let floatHandle = copy(of: float.handle)
        let intHandle = copy(of: int.handle)

        let expectedMixed = Mixed(_handles: [floatHandle, intHandle])

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

        let wHandle = copy(of: w.handle)
        let bHandle = copy(of: b.handle)
        let floatHandle = copy(of: float.handle)
        let intHandle = copy(of: int.handle)

        let expectedNested = Nested(
            _handles: [wHandle, bHandle, floatHandle, intHandle])
        
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

        let wHandle = copy(of: w.handle)
        let bHandle = copy(of: b.handle)
        let floatHandle = copy(of: float.handle)
        let intHandle = copy(of: int.handle)

        let expectedGeneric = Generic<Simple, Mixed>(
            _handles: [wHandle, bHandle, floatHandle, intHandle])

        XCTAssertEqual(expectedGeneric, generic)
    }

    func testNestedGenericTypeList() {
        struct NestedGeneric {
            func function() {
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
                let w = Tensor<Float>(0.1)
                let b = Tensor<Float>(0.1)
                let simple = Simple(w: w, b: b)
                let float = Tensor<Float>(0.1)
                let int = Tensor<Int32>(1)
                let mixed = Mixed(float: float, int: int)
                let genericSM = Generic<Simple, Mixed>(t: simple, u: mixed)
                let genericMS = Generic<Mixed, Simple>(t: mixed, u: simple)
                let generic = UltraNested(a: genericSM, b: genericMS)

                let wHandle1 = copy(of: w.handle)
                let wHandle2 = copy(of: w.handle)
                let bHandle1 = copy(of: b.handle)
                let bHandle2 = copy(of: b.handle)
                let floatHandle1 = copy(of: float.handle)
                let floatHandle2 = copy(of: float.handle)
                let intHandle1 = copy(of: int.handle)
                let intHandle2 = copy(of: int.handle)

                let expectedGeneric = UltraNested<Simple, Mixed>(
                    _handles: [wHandle1, bHandle1, floatHandle1,  intHandle1,
                        floatHandle2, intHandle2, wHandle2, bHandle2])

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
