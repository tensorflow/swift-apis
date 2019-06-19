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

final class LazyTensorTFFunctionBuilderTests : XCTestCase {
    override class func setUp() {
        super.setUp()
        _RuntimeConfig.useLazyTensor = true
    }

    func testSingletonInputs() {
        let a = materializedLazyTensor(Tensor<Float>(10.0))
        let w = Raw.identity(a)
        XCTAssertEqual(tfFunction(w, "testSingletonInputs")!.description,
            """

            testSingletonInputs(placeholder_0:float) -> (identity_1:float) {
              Identity_1 = Identity[T=float](placeholder_0)
              return identity_1 = Identity_1:output:0
            }

            """)
    }

    func testListInputs() {
        let a = materializedLazyTensor(Tensor<Float>(10.0))
        let b = materializedLazyTensor(Tensor<Float>(2.0))
        let w = Raw.addN(inputs: [a, b])
        XCTAssertEqual(tfFunction(w, "testListInputs")!.description,
            """

            testListInputs(placeholder_0:float, placeholder_1:float) -> (addn_2:float) {
              AddN_2 = AddN[N=2, T=float](placeholder_0, placeholder_1)
              return addn_2 = AddN_2:sum:0
            }

            """)
    }

    func testSequence() {
        let a = materializedLazyTensor(Tensor<Float>(10.0))
        let b = materializedLazyTensor(Tensor<Float>(2.0))
        let c = materializedLazyTensor(Tensor<Float>(3.0))
        let w = a + b * c
        XCTAssertEqual(tfFunction(w, "sequence")!.description,
            """

            sequence(placeholder_0:float, placeholder_1:float, placeholder_2:float) -> (add_4:float) {
              Mul_3 = Mul[T=float](placeholder_1, placeholder_2)
              Add_4 = Add[T=float](placeholder_0, Mul_3:z:0)
              return add_4 = Add_4:z:0
            }

            """)
    }

    func testAttributes() {
        // If tests ops such as "AttrBool" are available, testing the handing of
        // attributes would be very simple. However, tests ops are not
        // registered into the runtime by default (which is reasonable). If it
        // is possible to get a test-only libtensorflow.so, we should simplify
        // this test usinge the test ops.

        let a = materializedLazyTensor(Tensor<Float>(10.0))
        let b = materializedLazyTensor(Tensor<Float>(20.0))

        // Bool attribute
        let boolAttr = LazyTensorOperation("MatrixInverse", 1)
        boolAttr.updateAttribute("adjoint", true)
        boolAttr.addInput(a)
        XCTAssertEqual(tfFunction(boolAttr, "boolAttr").description,
            """

            boolAttr(placeholder_0:float) -> () {
              MatrixInverse_1 = MatrixInverse[T=float, adjoint=true](placeholder_0)
            }

            """)

        // Int attribute
        let intAttr = LazyTensorOperation("Unpack", 1)
        intAttr.updateAttribute("axis", 0)
        intAttr.updateAttribute("num", 1)
        intAttr.updateAttribute("T", Float.tensorFlowDataType)
        intAttr.addInput(a)
        XCTAssertEqual(tfFunction(intAttr, "intAttr").description,
            """

            intAttr(placeholder_0:float) -> () {
              Unpack_1 = Unpack[T=float, axis=0, num=1](placeholder_0)
            }

            """)

        // Float attribute
        let floatAttr = LazyTensorOperation("ApproximateEqual", 1)
        floatAttr.updateAttribute("T", Float.tensorFlowDataType)
        floatAttr.updateAttribute("tolerance", 0.01)
        floatAttr.addInput(a)
        floatAttr.addInput(b)
        XCTAssertEqual(tfFunction(floatAttr, "floatAttr").description,
            """

            floatAttr(placeholder_0:float, placeholder_1:float) -> () {
              ApproximateEqual_2 = ApproximateEqual[T=float, tolerance=0.01](placeholder_0, placeholder_1)
            }

            """)

        // String attribute
        let stringAttr = LazyTensorOperation("PrintV2", 0)
        let tag = StringTensor("Hello!")
        stringAttr.updateAttribute("output_stream", "stream")
        stringAttr.addInput(tag)
        XCTAssertEqual(tfFunction(stringAttr, "stringAttr").description,
            """

            stringAttr() -> () {
              Const_0 = Const[dtype=string, value=Tensor<type: string shape: [] values: Hello!>]()
              PrintV2_1 = PrintV2[end=\"\\n\", output_stream=\"stream\"](Const_0:output:0)
            }

            """)


        // TensorShape attr
        let shapeAttr = LazyTensorOperation("EnsureShape", 1)
        shapeAttr.updateAttribute("shape", TensorShape([5, 6]))
        shapeAttr.updateAttribute("T", Float.tensorFlowDataType)
        shapeAttr.addInput(a)
        XCTAssertEqual(tfFunction(shapeAttr, "shapeAttr").description,
            """

            shapeAttr(placeholder_0:float) -> () {
              EnsureShape_1 = EnsureShape[T=float, shape=[5,6]](placeholder_0)
            }

            """)


        // [Int], [TensorShape?] & [TensorDataType] attribute.
        let arrayAttr1 = LazyTensorOperation("PrelinearizeTuple", 0)
        arrayAttr1.updateAttribute("dtypes",
            [Float.tensorFlowDataType, Float.tensorFlowDataType]) // [TensorDataType]
        arrayAttr1.updateAttribute("shapes", [[1, 2], nil]) // [TensorShape?]
        arrayAttr1.updateAttribute("layouts", [3, 4]) // [Int]
        arrayAttr1.addInputList([a,b])

        XCTAssertEqual(tfFunction(arrayAttr1, "arrayAttr1").description,
            """

            arrayAttr1(placeholder_0:float, placeholder_1:float) -> () {
              PrelinearizeTuple_2 = PrelinearizeTuple[dtypes={float, float}, layouts=[3, 4], shapes=[[1,2], <unknown>]](placeholder_0, placeholder_1)
            }

            """)

        // Const Tensor attribute.
        let constTensorAttr = LazyTensorOperation("Const", 0)
        let x = Tensor<Float>(5.5)
        constTensorAttr.updateAttribute("dtype", Float.tensorFlowDataType)
        constTensorAttr.updateAttribute("value", x.handle.handle._tfeTensorHandle)
        XCTAssertEqual(tfFunction(constTensorAttr, "constTensorAttr").description,
            """

            constTensorAttr() -> () {
              Const_0 = Const[dtype=float, value=Tensor<type: float shape: [] values: 5.5>]()
            }

            """)

    }

    private func tfFunction(
        _ lazyOp: LazyTensorOperation,
        _ name: String? = nil
    ) -> TFFunction {
        let trace = LazyTensorTrace(lazyOp)
        return TFFunction(trace, name: name)
    }

    private func materializedLazyTensor<T: TensorFlowScalar>(
        _ input: Tensor<T>
    ) -> Tensor<T> {
        let concreteHandle = input.handle.handle._tfeTensorHandle
        let materializedHandle = LazyTensor(_materialized: concreteHandle)
        return Tensor(handle: TensorHandle<T>(handle: materializedHandle))
    }

    private func tfFunction<T: TensorFlowScalar>(
        _ input: Tensor<T>,
        _ name: String? = nil
    ) -> TFFunction? {
        let tensor = input.handle.handle
        guard let lazyTensor = tensor as? LazyTensor else {
            XCTFail("Trying to get TFFunction for a non-lazy tensor.")
            return nil
        }
        guard case let .symbolic(lazyOp, _, _)  = lazyTensor.handle else {
            XCTFail("Cannot get TFFunction for a concrete tensor.")
            return nil
        }
        let trace =  LazyTensorTrace(lazyOp)
        return TFFunction(trace, name: name)
    }

    static var allTests = [
        ("testSingletonInputs", testSingletonInputs),
        ("testListInputs", testListInputs),
        ("testSequence", testSequence),
        ("testAttributes", testAttributes),
    ]
}
