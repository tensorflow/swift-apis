import XCTest

@testable import TensorFlow
import CTensorFlow

final class LazyTensorTraceTests: XCTestCase {
    override class func setUp() {
        super.setUp()
        _RuntimeConfig.useLazyTensor = true
    }

    func testSingleLiveTensor() {
        let a = Tensor<Float>(10.0)
        let b = Tensor<Float>(2.0)
        let c = Tensor<Float>(3.0)
        let w = a + b * c
        XCTAssertEqual("\(lazyTrace(w)!)",
            """
            lazyTrace_5() -> (Add_4) {
              // operationsCount: 5
              Const_0[dtype: float, value: 10.0]():1
              Const_1[dtype: float, value: 2.0]():1
              Const_2[dtype: float, value: 3.0]():1
              Mul_3[T: float](Const_1:0, Const_2:0):1
              Add_4[T: float](Const_0:0, Mul_3:0):1
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
        XCTAssertEqual("\(lazyTrace(z)!)",
            """
            lazyTrace_8() -> (Add_4,Mul_5,Div_7) {
              // operationsCount: 8
              Const_0[dtype: float, value: 10.0]():1
              Const_1[dtype: float, value: 2.0]():1
              Add_2[T: float](Const_0:0, Const_1:0):1
              Const_3[dtype: float, value: 3.0]():1
              Add_4[T: float](Add_2:0, Const_3:0):1
              Mul_5[T: float](Add_4:0, Const_3:0):1
              Sub_6[T: float](Add_4:0, Const_3:0):1
              Div_7[T: float](Mul_5:0, Sub_6:0):1
            }
            """)

        // Note that we only pick operations on which the lazy tensor in
        // question depends on.
        XCTAssertEqual("\(lazyTrace(y)!)",
            """
            lazyTrace_6() -> (Add_4,Mul_5) {
              // operationsCount: 6
              Const_0[dtype: float, value: 10.0]():1
              Const_1[dtype: float, value: 2.0]():1
              Add_2[T: float](Const_0:0, Const_1:0):1
              Const_3[dtype: float, value: 3.0]():1
              Add_4[T: float](Add_2:0, Const_3:0):1
              Mul_5[T: float](Add_4:0, Const_3:0):1
            }
            """)
    }

    func testSimpleControlFlow() {
        let a = Tensor<Float>(5.0)
        let addOrMul = { (useAdd: Bool, a: Tensor<Float>) in
            useAdd ? (a + a) : (a * a)
        }
        let add = addOrMul(/*useAdd:*/true, a)
        XCTAssertEqual("\(lazyTrace(add)!)",
            """
            lazyTrace_2() -> (Add_1) {
              // operationsCount: 2
              Const_0[dtype: float, value: 5.0]():1
              Add_1[T: float](Const_0:0, Const_0:0):1
            }
            """)
        let mul = addOrMul(/*useAdd:*/false, a)
        XCTAssertEqual("\(lazyTrace(mul)!)",
            """
            lazyTrace_2() -> (Mul_1) {
              // operationsCount: 2
              Const_0[dtype: float, value: 5.0]():1
              Mul_1[T: float](Const_0:0, Const_0:0):1
            }
            """)
    }

    private func lazyTrace<T: TensorFlowScalar>(
        _ input: Tensor<T>) -> LazyTensorTrace? {
        let tensor = input.handle.handle
        guard let lazyTensor = tensor as? LazyTensor else {
            XCTFail("Trying to get lazy trace for a non-lazy tensor.")
            return nil
        }
        guard case let
            LazyTensor.Handle.symbolic(lazyOp, _, _)  = lazyTensor.handle else {
            XCTFail("Cannot get lazy trace for a concrete tensor.")
            return nil
        }
        return LazyTensorTrace(lazyOp)
    }

    static var allTests = [
        ("testSingleLiveTensor", testSingleLiveTensor),
        ("testMultipleLiveTensors", testMultipleLiveTensors),
        ("testSimpleControlFlow", testSimpleControlFlow)
    ]
}
