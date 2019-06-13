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

let cube: @differentiable (Tensor<Float>) -> Tensor<Float> = { ($0 * $0 * $0) }

@differentiable(vjp: vjpFoo)
func foo(_ x: Tensor<Float>) -> Tensor<Float> {
    return Raw.identity(x)
}
func vjpFoo(_ x: Tensor<Float>) -> (Tensor<Float>, (Tensor<Float>) -> Tensor<Float>) {
    return (foo(x), { v in v })
}

final class TensorAutoDiffTests: XCTestCase {
    func testSimpleGrad() {
        func square(_ x: Tensor<Float>) -> Tensor<Float> {
            return (x * x).sum()
        }
        XCTAssertEqual([0.2, 0.4, 0.6], gradient(at: [0.1, 0.2, 0.3], in: square))
        XCTAssertEqual([[20], [40]], gradient(at: [[10], [20]], in: square))
    }

    func testGenericGrad() {
        func square<T : TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
            return (x * x).sum()
        }
        XCTAssertEqual([0.2, 0.4, 0.6], gradient(at: Tensor([0.1, 0.2, 0.3]), in: square))
    }

    func testScalarGenericGrad() {
        // Tests TF-287.
        func negate<T : TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
            return (1 - x).sum()
        }
        XCTAssertEqual(Tensor(-1), gradient(at: Tensor([0.1, 0.2, 0.3]), in: negate))
    }

    func testScalarized() {
        let grad = gradient(at: Tensor<Float>([3.0, 4.0])) { x in
            logSoftmax(x).mean().scalarized()
        }
        XCTAssertEqual(Tensor([0.23105857, -0.2310586]), grad)
    }

    func testPlus() {
        func f(a: Tensor<Float>, b: Tensor<Float>) -> Tensor<Float> { a + b }
        XCTAssertTrue((Tensor(1), Tensor(1)) == gradient(at: Tensor(0), Tensor(0), in: f))
        XCTAssertTrue(([1], [1]) == pullback(at: [1], [10], in: f)([1]))
    }

    func testSubtract() {
        func f(a: Tensor<Float>, b: Tensor<Float>) -> Tensor<Float> { a - b }
        XCTAssertTrue((Tensor(1), Tensor(-1)) == gradient(at: Tensor(0), Tensor(0), in: f))
        XCTAssertTrue(([1], [-1]) == pullback(at: [1], [10], in: f)([1]))
    }
    
    func testMultiply() {
        func f(a: Tensor<Float>, b: Tensor<Float>) -> Tensor<Float> { (a * b).sum() }
        XCTAssertTrue(([0], [0]) == gradient(at: [0], [0], in: f))
        XCTAssertTrue(([10], [1]) == gradient(at: [1], [10], in: f))
    }

    func testDivide() {
        func f(a: Tensor<Float>, b: Tensor<Float>) -> Tensor<Float> { a / b }
        XCTAssertTrue(([0.1], [-0.01]) == pullback(at: [1], [10], in: f)([1]))
    }

    func testMatmul() {
        func f(a: Tensor<Float>, b: Tensor<Float>) -> Tensor<Float> { matmul(a, b) }
        let v = Tensor<Float>(ones: [1, 1])
        XCTAssertTrue(([[0]], [[0]]) == pullback(at: [[0]], [[0]], in: f)(v))
        XCTAssertTrue(([[10]], [[1]]) == pullback(at: [[1]], [[10]], in: f)(v))
    }

    func testDot() {
        func f(a: Tensor<Float>, b: Tensor<Float>) -> Tensor<Float> { a • b }
        let v = Tensor<Float>(ones: [1, 1])
        XCTAssertTrue(([[0]], [[0]]) == pullback(at: [[0]], [[0]], in: f)(v))
        XCTAssertTrue(([[10]], [[1]]) == pullback(at: [[1]], [[10]], in: f)(v))
    }

    func testNegate() {
        func f(a: Tensor<Float>) -> Tensor<Float> { (-a).sum() }
        XCTAssertEqual([-1], gradient(at: [0], in: f))
        XCTAssertEqual([-1], gradient(at: [10], in: f))
    }

    func testAbs() {
        func f(a: Tensor<Float>) -> Tensor<Float> { abs(a).sum() }
        XCTAssertEqual([1, -1, 0], gradient(at: [3.0, -3.0, 0], in: f))
    }

    func testSum() {
        let input = Tensor<Float>(repeating: 42, shape: [2, 2])
        let sumPullbackScalar = pullback(at: input) { (a: Tensor<Float>) in a.sum() }
        let sumPullbackSqueezingAxes = pullback(at: input) { (a: Tensor<Float>) in
            a.sum(squeezingAxes: 0, 1)
        }
        let sumPullbackAlongAxes = pullback(at: input) { (a: Tensor<Float>) in
            a.sum(alongAxes: 0, 1)
        }

        let expected = Tensor<Float>(ones: [2, 2])
        XCTAssertEqual(expected, sumPullbackScalar(Tensor(1)))
        XCTAssertEqual(expected, sumPullbackSqueezingAxes(Tensor(1)))
        XCTAssertEqual(expected, sumPullbackAlongAxes(Tensor(1)))
        XCTAssertEqual(expected * 3, sumPullbackScalar(Tensor(3)))
        XCTAssertEqual(expected * 3, sumPullbackSqueezingAxes(Tensor(3)))
        XCTAssertEqual(expected * 3, sumPullbackAlongAxes(Tensor(3)))
    }

    func testMean() {
        let meanGradScalar = gradient { (a: Tensor<Float>) in a.mean().sum() }
        let meanGradSqueezingAxes = gradient { (a: Tensor<Float>) in
            a.mean(squeezingAxes: 0, 1).sum()
        }
        let meanGradAlongAxes = gradient { (a: Tensor<Float>) in a.mean(alongAxes: 0, 1).sum() }

        let input = Tensor<Float>(ones: [2, 2])
        let expected = Tensor<Float>(repeating: 0.25, shape: [2, 2])
        XCTAssertEqual(expected, meanGradScalar(input))
        XCTAssertEqual(expected, meanGradSqueezingAxes(input))
        XCTAssertEqual(expected, meanGradAlongAxes(input))
    }

    func testVariance() {
        let varianceGradScalar = gradient { (a: Tensor<Float>) in a.variance().sum() }
        let varianceGradSqueezingAxes = gradient { (a: Tensor<Float>) in
            a.variance(squeezingAxes: 0, 1).sum()
        }
        let varianceGradAlongAxes = gradient { (a: Tensor<Float>) in
            a.variance(alongAxes: 0, 1).sum()
        }

        let input: Tensor<Float> = [[1, 2], [3, 4]]
        let expected: Tensor<Float> = [[-0.75, -0.25], [0.25, 0.75]]
        XCTAssertEqual(expected, varianceGradScalar(input))
        XCTAssertEqual(expected, varianceGradSqueezingAxes(input))
        XCTAssertEqual(expected, varianceGradAlongAxes(input))
    }

    func testExpandingShape() {
        func f1(a: Tensor<Float>) -> Tensor<Float> { a.expandingShape(at: 0).squared() }
        func f2(a: Tensor<Float>) -> Tensor<Float> { a.squared().expandingShape(at: 0) }
        XCTAssertEqual([6, 10], pullback(at: [3, 5], in: f1)([[1, 1]]))
        XCTAssertEqual([6, 10], pullback(at: [3, 5], in: f2)([[1, 1]]))
    }

    func testSqueezingShape() {
        func f1(a: Tensor<Float>) -> Tensor<Float> { a.squeezingShape(at: 0).squared() }
        func f2(a: Tensor<Float>) -> Tensor<Float> { a.squared().squeezingShape(at: 0) }
        XCTAssertEqual([[6, 10]], pullback(at: [[3, 5]], in: f1)([1, 1]))
        XCTAssertEqual([[6, 10]], pullback(at: [[3, 5]], in: f2)([1, 1]))
    }

    func testReshapedBackprop() {
        func f1(a: Tensor<Float>) -> Tensor<Float> { a.reshaped(toShape: Tensor<Int32>([2, 1])).squared() }
        func f2(a: Tensor<Float>) -> Tensor<Float> { a.squared().reshaped(toShape: Tensor<Int32>([2, 1])) }
        XCTAssertEqual([[6, 10]], pullback(at: [[3, 5]], in: f1)([[1], [1]]))
        XCTAssertEqual([[6, 10]], pullback(at: [[3, 5]], in: f2)([[1], [1]]))
    }

    func testReshaped() {
        let shapeTensor = Tensor<Int32>([2, 2, 2])
        let input = Tensor<Float>(ones: [2, 4])
        let reshapedPullback = pullback(at: input) { (a: Tensor<Float>) in
            a.reshaped(toShape: shapeTensor)
        }
        let reshaped = Tensor<Float>(ones: [2, 2, 2])
        XCTAssertEqual(input, reshapedPullback(reshaped))
    }

    func testConcatenationPlusPlus() {
        let a1 = Tensor<Float>([1,2,3,4])
        let b1 = Tensor<Float>([5,6,7,8,9,10])

        let a2 = Tensor<Float>([1,1,1,1])
        let b2 = Tensor<Float>([1,1,1,1,1,1])

        let grads = gradient(at: a2, b2) { a, b in
            return ((a1 * a) ++ (b1 * b)).sum()
        }

        XCTAssertEqual(a1, grads.0)
        XCTAssertEqual(b1, grads.1)
    }

    func testConcatenated() {
        let a1 = Tensor<Float>([1,2,3,4])
        let b1 = Tensor<Float>([5,6,7,8,9,10])

        let a2 = Tensor<Float>([1,1,1,1])
        let b2 = Tensor<Float>([1,1,1,1,1,1])

        let grads = gradient(at: a2, b2) { a, b in
            return (a1 * a).concatenated(with: b1 * b, alongAxis: -1).sum()
        }

        XCTAssertEqual(a1, grads.0)
        XCTAssertEqual(b1, grads.1)
    }

    func testTransposed() {
        let input = Tensor<Float>(ones: [2, 3])
        let transposed = Tensor<Float>(ones: [3, 2])
        let transposedPullback = pullback(at: input) { (a: Tensor<Float>) in a.transposed() }
        let transposedPermutationsPullback = pullback(at: input) { (a: Tensor<Float>) in
            a.transposed(withPermutations: [1, 0])
        }
        let transposedVariadicsPullback = pullback(at: input) { (a: Tensor<Float>) in
            a.transposed(withPermutations: 1, 0)
        }

        XCTAssertEqual(input, transposedPullback(transposed))
        XCTAssertEqual(input, transposedPermutationsPullback(transposed))
        XCTAssertEqual(input, transposedVariadicsPullback(transposed))
    }

    func testRelu() {
        func f(a: Tensor<Float>) -> Tensor<Float> { relu(a).sum() }
        XCTAssertEqual([1, 0, 0], gradient(at: [5, -5, 0], in: f))
    }

    func testSoftmax() {
        let pb = pullback(at: Tensor(ones: [2, 2])) { (a: Tensor<Float>) in softmax(a) }
        XCTAssertEqual([[0, 0], [0, 0]], pb([[1, 1], [1, 1]]))
        XCTAssertEqual([[-0.25, 0.25], [0.75, -0.75]], pb([[1, 2], [4, 1]]))
    }

    func testLogSoftmax() {
        let pb = pullback(at: Tensor(ones: [3, 3])) { (a: Tensor<Float>) in logSoftmax(a) }
        XCTAssertEqual(Tensor(repeating: 5.9604645e-08, shape: [3, 3]), pb(Tensor(ones: [3, 3])))
    }

    // SR-9345
    func testOwnedCheckpoints() {
        func body(_ x: Tensor<Float>) -> Tensor<Float> {
            return foo(foo(x))
        }
        
        let pb = pullback(at: Tensor(Float(10)), in: body)
        XCTAssertEqual(Tensor(1), pb(Tensor(1)))
    }

    // SR-9804
    func testADRefcounting() {
        func f(_ x: Tensor<Float>) -> Tensor<Float> {
            return x
        }
        XCTAssertEqual(Tensor(1), gradient(at: Tensor(0), in: f))
    }

    func testDifferentiateGlobal() {
        XCTAssertEqual(Tensor(48), gradient(at: Tensor(4), in: cube))
    }

    func testSideEffects() {
        let foo: @differentiable (Tensor<Float>) -> Tensor<Float> = { x in
            var a = x
            a = a + x
            a = a + x
            return a + x
        }
        XCTAssertEqual(Tensor([4, 4]), pullback(at: Tensor([4, 5]), in: foo)([1, 1]))

        func bar(x: Tensor<Float>) -> Tensor<Float> {
            var a = x
            a = a * x
            a = a * x
            return a.sum()
        }
        XCTAssertEqual(Tensor(48), gradient(at: Tensor(4), in: bar))
    }

    func testBroadcastToShape() {
        func foo(tensor: Tensor<Float>, shape: Tensor<Int32>) -> Tensor<Float> {
            tensor.broadcasted(toShape: shape)
        }

        let pb: (Tensor<Float>) -> Tensor<Float> = pullback(at: Tensor([99, 33, 55])) { x in
            foo(tensor: x, shape: Tensor([3, 3]))
        }
        let inputTensor: Tensor<Float> = Tensor([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]]
        )
        let expected: Tensor<Float> = Tensor([[4, 8, 12]])
        XCTAssertEqual(expected, pb(inputTensor))
    }

    func testBroadcastTo() {
        func foo(tensor: Tensor<Float>, shape: TensorShape) -> Tensor<Float> {
            tensor.broadcasted(to: shape)
        }
        let pb: (Tensor<Float>) -> Tensor<Float> = pullback(at: Tensor([99, 33, 55])) { x in
            foo(tensor: x, shape: TensorShape([3, 3]))
        }
        let inputTensor: Tensor<Float> = Tensor([1, 2, 3])
        let expected: Tensor<Float> = Tensor([1, 2, 3])
        XCTAssertEqual(expected, pb(inputTensor))
    }

    func testBroadcastLike() {
        func foo(tensor: Tensor<Float>, other: Tensor<Double>) -> Tensor<Float> {
            tensor.broadcasted(like: other)
        }
        let pb: (Tensor<Float>) -> Tensor<Float> = pullback(at: Tensor([99, 33, 55])) { x in
            foo(tensor: x, other: Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]]))
        }
        let inputTensor: Tensor<Float> = Tensor([[[[[[1, 2, 3]]]]]])
        let expected: Tensor<Float> = Tensor([[1, 2, 3]])

        XCTAssertEqual(expected, pb(inputTensor))
    }

    func testUnbroadcastToShape() {
        func foo(tensor: Tensor<Float>, shape: Tensor<Int32>) -> Tensor<Float> {
            tensor.unbroadcasted(toShape: shape)
        }
        let atTensor: Tensor<Float> = Tensor([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]]
        )
        let pb: (Tensor<Float>) -> Tensor<Float> = pullback(at: atTensor) { x in
            foo(tensor: x, shape: Tensor([1, 3]))
        }
        let expected = atTensor
        let inputTensor: Tensor<Float> = Tensor([[1, 2, 3]])
        XCTAssertEqual(expected, pb(inputTensor))
    }

    func testUnbroadcastTo() {
        func foo(tensor: Tensor<Float>, shape: TensorShape) -> Tensor<Float> {
            tensor.unbroadcasted(to: shape)
        }
        let atTensor: Tensor<Float> = Tensor([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]]
        )
        let pb: (Tensor<Float>) -> Tensor<Float> = pullback(at: atTensor) { x in
            foo(tensor: x, shape: TensorShape([1, 3]))
        }
        let inputTensor: Tensor<Float> = Tensor([2])
        let expected: Tensor<Float> = Tensor([
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2]]
        )
        XCTAssertEqual(expected, pb(inputTensor))
    }

    func testUnbroadcastLike() {
        func foo(tensor: Tensor<Float>, other: Tensor<Double>) -> Tensor<Float> {
            tensor.unbroadcasted(like: other)
        }
        let atTensor: Tensor<Float> = Tensor([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]]
        )
        let pb: (Tensor<Float>) -> Tensor<Float> = pullback(at: atTensor) { x in
            foo(tensor: x, other: Tensor([[1, 2, 3]]))
        }
        let inputTensor: Tensor<Float> = Tensor([
            [8, 1, 3],
            [8, 1, 3],
            [8, 1, 3]]
        )
        let expected: Tensor<Float> = inputTensor
        XCTAssertEqual(expected, pb(inputTensor))
    }
    
    static var allTests = [
        ("testSimpleGrad", testSimpleGrad),
        ("testGenericGrad", testGenericGrad),
        ("testScalarGenericGrad", testScalarGenericGrad),
        ("testScalarized", testScalarized),
        ("testPlus", testPlus),
        ("testSubtract", testSubtract),
        ("testMultiply", testMultiply),
        ("testDivide", testDivide),
        ("testMatmul", testMatmul),
        ("testDot", testDot),
        ("testNegate ", testNegate),
        ("testAbs", testAbs),
        ("testSum", testSum),
        ("testMean", testMean),
        ("testVariance", testVariance),
        ("testExpandingShape", testExpandingShape),
        ("testSqueezingShape", testSqueezingShape),
        ("testReshapedBackprop", testReshapedBackprop),
        ("testReshaped", testReshaped),
        ("testConcatenationPlusPlus", testConcatenationPlusPlus),
        ("testConcatenated", testConcatenated),
        ("testTransposed", testTransposed),
        ("testRelu", testRelu),
        ("testSoftmax", testSoftmax),
        ("testLogSoftmax", testLogSoftmax),
        ("testOwnedCheckpoints", testOwnedCheckpoints),
        ("testADRefcounting", testADRefcounting),
        ("testDifferentiateGlobal", testDifferentiateGlobal),
        ("testSideEffects", testSideEffects),
        ("testBroadcastToShape", testBroadcastToShape),
        ("testBroadcastTo", testBroadcastTo),
        ("testBroadcastLike", testBroadcastLike),
        ("testUnbroadcastToShape", testUnbroadcastToShape),
        ("testUnbroadcastTo", testUnbroadcastTo),
        ("testUnbroadcastLike", testUnbroadcastLike)
    ]
}
