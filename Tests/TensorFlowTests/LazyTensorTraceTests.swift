import XCTest

@testable import TensorFlow
import CTensorFlow

final class LazyTensorTraceTests: XCTestCase {
    override class func setUp() {
        super.setUp()
        _RuntimeConfig.useLazyTensor = true
    }

    func testStraightLine() {
        let a = Tensor<Float>(10.0)
        let b = Tensor<Float>(2.0)
        let c = Tensor<Float>(3.0)
        let w = a + b * c
        // let x = w - c
        // let y = x + x
        // let z = y + y
        let trace = lazyTrace(w)!
        print("\(trace)")
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
        ("testStraightLine", testStraightLine),
    ]
}
