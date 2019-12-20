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
    return _Raw.identity(x)
}
func vjpFoo(_ x: Tensor<Float>) -> (Tensor<Float>, (Tensor<Float>) -> Tensor<Float>) {
    return (foo(x), { v in v })
}

final class TensorAutoDiffTests: XCTestCase {
    func testSimpleGrad() {
        func square(_ x: Tensor<Float>) -> Tensor<Float> {
            return (x * x).sum()
        }
        XCTAssertEqual(gradient(at: [0.1, 0.2, 0.3], in: square), [0.2, 0.4, 0.6])
        XCTAssertEqual(gradient(at: [[10], [20]], in: square), [[20], [40]])
    }

    func testGenericGrad() {
        func square<T : TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
            return (x * x).sum()
        }
        XCTAssertEqual(gradient(at: Tensor([0.1, 0.2, 0.3]), in: square), [0.2, 0.4, 0.6])
    }

    func testConditionals() {
        func condNestedTupleVar(_ x: Tensor<Float>) -> Tensor<Float> {
            // Convoluted function returning `x + x`.
            var y: (Tensor<Float>, Tensor<Float>) = (x + x, x - x)
            var z: ((Tensor<Float>, Tensor<Float>), Tensor<Float>) = (y, x)
            if (x .> 0).all() {
                let w = (x, x)
                y.0 = w.1
                y.1 = w.0
                z.0.0 = z.0.0 - y.0
                z.0.1 = z.0.1 + y.0
            } else {
                z = ((y.0 - x, y.1 + x), x)
            }
            return y.0 + y.1 - z.0.0 + z.0.1
        }
        XCTAssertTrue((value: Tensor(8), gradient: Tensor(2)) == valueWithGradient(at: Tensor(4), in: condNestedTupleVar))
        XCTAssertTrue((value: Tensor(-20), gradient: Tensor(2)) == valueWithGradient(at: Tensor(-10), in: condNestedTupleVar))
        XCTAssertTrue((value: Tensor(-2674), gradient: Tensor(2)) == valueWithGradient(at: Tensor(-1337), in: condNestedTupleVar))

        func guard2Var(_ x: Tensor<Float>, _ y: Tensor<Float>) -> Tensor<Float> {
            var z = y
            guard (x .> 0).all() else {
                if (y .> 0).all() {
                    z = z * x
                } else if x == Tensor(-1337) {
                    z = x
                    z = z * z
                } else {
                    z = Tensor(0)
                }
                return z
            }
            return z * y
        }
        XCTAssertTrue((Tensor(0), Tensor(10)) == gradient(at: Tensor(4), Tensor(5), in: guard2Var))
        XCTAssertTrue((Tensor(5), Tensor(-1337)) == gradient(at: Tensor(-1337), Tensor(5), in: guard2Var))
        XCTAssertTrue((Tensor(-2674), Tensor(0)) == gradient(at: Tensor(-1337), Tensor(-5), in: guard2Var))
        XCTAssertTrue((Tensor(2), Tensor(-3)) == gradient(at: Tensor(-3), Tensor(2), in: guard2Var))
    }

    func testNestedConditionals() {
        // Test tensor-tensor ops.
        func condNested1(_ x: Tensor<Float>, _ y: Tensor<Float>) -> Tensor<Float> {
            if (x .> 0).all() {
                if (y .> 10).all() {
                    let z = x * y
                    if (z .> 100).all() {
                        return x + z
                    } else if y == Tensor(20) {
                        return z + z
                    }
                } else {
                    return x + y
                }
            }
            return -y
        }
        XCTAssertTrue((Tensor(40), Tensor(8)) == gradient(at: Tensor(4), Tensor(20), in: condNested1))
        XCTAssertTrue((Tensor(0), Tensor(-1)) == gradient(at: Tensor(4), Tensor(21), in: condNested1))
        XCTAssertTrue((Tensor(1), Tensor(1)) == gradient(at: Tensor(4), Tensor(5), in: condNested1))
        XCTAssertTrue((Tensor(0), Tensor(-1)) == gradient(at: Tensor(-3), Tensor(-2), in: condNested1))

        // Test tensor-scalar ops.
        func condNested2(_ x: Tensor<Float>, _ y: Float) -> Tensor<Float> {
            if (x .> 0).all() {
                if y > 10 {
                    let z = x * y
                    if (z .> 100).all() {
                        return x + z
                    } else if y == 20 {
                        return z + z
                    }
                } else {
                    return x + y
                }
            }
            return Tensor(-y)
        }
        XCTAssertTrue((Tensor(40), 8) == gradient(at: Tensor(4), 20, in: condNested2))
        XCTAssertTrue((Tensor(0), -1) == gradient(at: Tensor(4), 21, in: condNested2))
        XCTAssertTrue((Tensor(1), 1) == gradient(at: Tensor(4), 5, in: condNested2))
        XCTAssertTrue((Tensor(0), -1) == gradient(at: Tensor(-3), -2, in: condNested2))
    }

    func testRecursion() {
        func factorial(_ x: Tensor<Float>) -> Tensor<Float> {
            if x == Tensor(1) {
                return Tensor(1)
            }
            return x * factorial(x - 1)
        }
        XCTAssertEqual(gradient(at: Tensor(1), in: factorial), Tensor(0))
        XCTAssertEqual(gradient(at: Tensor(2), in: factorial), Tensor(1))
        XCTAssertEqual(gradient(at: Tensor(3), in: factorial), Tensor(5))
        XCTAssertEqual(gradient(at: Tensor(4), in: factorial), Tensor(26))
        XCTAssertEqual(gradient(at: Tensor(5), in: factorial), Tensor(154))

        func product(_ x: Tensor<Float>, count: Int) -> Tensor<Float> {
            precondition(count > 0)
            if count == 1 {
                return x
            }
            return x * product(x, count: count - 1)
        }
        XCTAssertEqual(gradient(at: Tensor(-10), in: { x in product(x, count: 2) }), Tensor(-20))
        XCTAssertEqual(gradient(at: Tensor(10), in: { x in product(x, count: 3) }), Tensor(300))
        XCTAssertEqual(gradient(at: Tensor(100), in: { x in product(x, count: 1) }), Tensor(1))
    }

    func testScalarGenericGrad() {
        // Tests TF-287.
        func negate<T : TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
            return (1 - x).sum()
        }
        XCTAssertEqual(gradient(at: Tensor([0.1, 0.2, 0.3]), in: negate), Tensor([-1, -1, -1]))
    }

    func testScalarized() {
        let grad = gradient(at: Tensor<Float>([3.0, 4.0])) { x in
            logSoftmax(x).mean().scalarized()
        }
        XCTAssertEqual(grad, Tensor([0.23105857, -0.2310586]))
    }

    func testScalars() {
        let grad = gradient(at: Tensor<Float>([3, 4])) { x in
            x.scalars.differentiableReduce(0, { $0 + $1 })
        }
        XCTAssertEqual(grad, Tensor([1, 1]))
    }

    func testInitFromScalars() {
        let grad = gradient(at: [3.0, 4.0]) { x in
            Tensor(x).sum()
        }
        XCTAssertEqual(grad, Array<Double>.TangentVector([1, 1]))
    }

    func testInitFromScalarsWithShape() {
        let grad = gradient(at: [3.0, 4.0]) { x in
            Tensor(shape: [1, 2, 1, 1], scalars: x).sum()
        }
        XCTAssertEqual(grad, Array<Double>.TangentVector([1, 1]))
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
        XCTAssertEqual(gradient(at: [0], in: f), [-1])
        XCTAssertEqual(gradient(at: [10], in: f), [-1])
    }

    func testAbs() {
        func f(a: Tensor<Float>) -> Tensor<Float> { abs(a).sum() }
        XCTAssertEqual(gradient(at: [3.0, -3.0, 0], in: f), [1, -1, 0])
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
        XCTAssertEqual(sumPullbackScalar(Tensor(1)), expected)
        XCTAssertEqual(sumPullbackSqueezingAxes(Tensor(1)), expected)
        XCTAssertEqual(sumPullbackAlongAxes(Tensor(1)), expected)
        XCTAssertEqual(sumPullbackScalar(Tensor(3)), expected * 3)
        XCTAssertEqual(sumPullbackSqueezingAxes(Tensor(3)), expected * 3)
        XCTAssertEqual(sumPullbackAlongAxes(Tensor(3)), expected * 3)
    }

    func testMean() {
        let meanGradScalar = gradient { (a: Tensor<Float>) in a.mean().sum() }
        let meanGradSqueezingAxes = gradient { (a: Tensor<Float>) in
            a.mean(squeezingAxes: 0, 1).sum()
        }
        let meanGradAlongAxes = gradient { (a: Tensor<Float>) in a.mean(alongAxes: 0, 1).sum() }

        let input = Tensor<Float>(ones: [2, 2])
        let expected = Tensor<Float>(repeating: 0.25, shape: [2, 2])
        XCTAssertEqual(meanGradScalar(input), expected)
        XCTAssertEqual(meanGradSqueezingAxes(input), expected)
        XCTAssertEqual(meanGradAlongAxes(input), expected)
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
        XCTAssertEqual(varianceGradScalar(input), expected)
        XCTAssertEqual(varianceGradSqueezingAxes(input), expected)
        XCTAssertEqual(varianceGradAlongAxes(input), expected)
    }

    func testMin() {
        // The expected gradient values were computed using the following TensorFlow 2.0 Beta1
        // Python code with respective `a` and `b` tensors:
        // ```
        // with tf.GradientTape() as t:
        //     t.watch([a, b])
        //     y = tf.math.reduce_sum(tf.minimum(a, b))
        // print(t.gradient(y, [a, b]))
        // ```
        do {
            let a = Tensor<Float>([4, 5, 3])
            let b = Tensor<Float>([4, 2, 6])
            let computedGradient1 = gradient(at: a, b) { a, b in min(a, b).sum() }
            let expectedGradient1: (Tensor<Float>, Tensor<Float>) = (
                [1.0, 0.0, 1.0], [0.0, 1.0, 0.0])
            XCTAssertEqual(computedGradient1.0, expectedGradient1.0)
            XCTAssertEqual(computedGradient1.1, expectedGradient1.1)

            let computedGradient2 = gradient(at: a, b) { a, b in min(b, a).sum() }
            let expectedGradient2: (Tensor<Float>, Tensor<Float>) =  (
                [0.0, 0.0, 1.0], [1.0, 1.0, 0.0])
            XCTAssertEqual(computedGradient2.0, expectedGradient2.0)
            XCTAssertEqual(computedGradient2.1, expectedGradient2.1)
        }

        do {
            let a = Tensor<Float>([[3.0, -2.0], [0.3, 10.0]])
            let b = Tensor<Float>([9.0, -3.0])
            let computedGradient = gradient(at: a, b) { a, b in min(a, b).sum() }
            let expectedGradient: (Tensor<Float>, Tensor<Float>) = (
                [[1.0, 0.0], [1.0, 0.0]], [0.0, 2.0])
            XCTAssertEqual(computedGradient.0, expectedGradient.0)
            XCTAssertEqual(computedGradient.1, expectedGradient.1)
        }
    }

    func testMax() {
        // The expected gradient values were computed using the following TensorFlow 2.0 Beta1
        // Python code with respective `a` and `b` tensors:
        // ```
        // with tf.GradientTape() as t:
        //     t.watch([a, b])
        //     y = tf.math.reduce_sum(tf.maximum(a, b))
        // print(t.gradient(y, [a, b]))
        // ```
        do {
            let a = Tensor<Float>([4, 5, 3])
            let b = Tensor<Float>([4, 2, 6])
            let computedGradient1 = gradient(at: a, b) { a, b in max(a, b).sum() }
            let expectedGradient1: (Tensor<Float>, Tensor<Float>) = (
                [1.0, 1.0, 0.0], [0.0, 0.0, 1.0])
            XCTAssertEqual(computedGradient1.0, expectedGradient1.0)
            XCTAssertEqual(computedGradient1.1, expectedGradient1.1)

            let computedGradient2 = gradient(at: a, b) { a, b in max(b, a).sum() }
            let expectedGradient2: (Tensor<Float>, Tensor<Float>) = (
                [0.0, 1.0, 0.0],  [1.0, 0.0, 1.0])
            XCTAssertEqual(computedGradient2.0, expectedGradient2.0)
            XCTAssertEqual(computedGradient2.1, expectedGradient2.1)
        }
        do {
            let a = Tensor<Float>([[3.0, -2.0], [0.3, 10.0]])
            let b = Tensor<Float>([9.0, -3.0])
            let computedGradient = gradient(at: a, b) { a, b in max(a, b).sum() }
            let expectedGradient: (Tensor<Float>, Tensor<Float>)  = (
                [[0.0, 1.0], [0.0, 1.0]], [2.0, 0.0])
            XCTAssertEqual(computedGradient.0, expectedGradient.0)
            XCTAssertEqual(computedGradient.1, expectedGradient.1)
        }
    }

    func testTensorInitStacking() {
        let a1 = Tensor<Float>([1, 2, 3, 4, 5])
        let b1 = Tensor<Float>([6, 7, 8, 9, 10])
        let a2 = Tensor<Float>([1, 1, 1, 1, 1])
        let b2 = Tensor<Float>([1, 1, 1, 1, 1])
        let grads = gradient(at: a2, b2) { a, b in
            Tensor<Float>(stacking: [a1 * a, b1 * b], alongAxis: -1).sum()
        }
        XCTAssertEqual(a1, grads.0)
        XCTAssertEqual(b1, grads.1)
    }

    func testExpandingShape() {
        func f1(a: Tensor<Float>) -> Tensor<Float> { a.expandingShape(at: 0).squared() }
        func f2(a: Tensor<Float>) -> Tensor<Float> { a.squared().expandingShape(at: 0) }
        XCTAssertEqual(pullback(at: [3, 5], in: f1)([[1, 1]]), [6, 10])
        XCTAssertEqual(pullback(at: [3, 5], in: f2)([[1, 1]]), [6, 10])
    }

    func testSqueezingShape() {
        func f1(a: Tensor<Float>) -> Tensor<Float> { a.squeezingShape(at: 0).squared() }
        func f2(a: Tensor<Float>) -> Tensor<Float> { a.squared().squeezingShape(at: 0) }
        XCTAssertEqual(pullback(at: [[3, 5]], in: f1)([1, 1]), [[6, 10]])
        XCTAssertEqual(pullback(at: [[3, 5]], in: f2)([1, 1]), [[6, 10]])
    }

    func testReshapedBackprop() {
        func f1(a: Tensor<Float>) -> Tensor<Float> { a.reshaped(toShape: Tensor<Int32>([2, 1])).squared() }
        func f2(a: Tensor<Float>) -> Tensor<Float> { a.squared().reshaped(toShape: Tensor<Int32>([2, 1])) }
        XCTAssertEqual(pullback(at: [[3, 5]], in: f1)([[1], [1]]), [[6, 10]])
        XCTAssertEqual(pullback(at: [[3, 5]], in: f2)([[1], [1]]), [[6, 10]])
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
            a.transposed(permutation: [1, 0])
        }
        let transposedVariadicsPullback = pullback(at: input) { (a: Tensor<Float>) in
            a.transposed(permutation: 1, 0)
        }

        XCTAssertEqual(input, transposedPullback(transposed))
        XCTAssertEqual(input, transposedPermutationsPullback(transposed))
        XCTAssertEqual(input, transposedVariadicsPullback(transposed))
    }

    func testSigmoid() {
        func f(a: Tensor<Float>) -> Tensor<Float> { sigmoid(a).sum() }
        assertEqual(gradient(at: [-1, 0, 1], in: f), [0.1966119, 0.25, 0.1966119], accuracy: 0.0001)
    }

    func testRelu() {
        func f(a: Tensor<Float>) -> Tensor<Float> { relu(a).sum() }
        XCTAssertEqual(gradient(at: [5, -5, 0], in: f), [1, 0, 0])
    }

    func testSoftmax() {
        let pb = pullback(at: Tensor(ones: [2, 2])) { (a: Tensor<Float>) in softmax(a) }
        XCTAssertEqual(pb([[1, 1], [1, 1]]), [[0, 0], [0, 0]])
        XCTAssertEqual(pb([[1, 2], [4, 1]]), [[-0.25, 0.25], [0.75, -0.75]])
    }

    func testLogSoftmax() {
        let pb = pullback(at: Tensor(ones: [3, 3])) { (a: Tensor<Float>) in logSoftmax(a) }
        XCTAssertEqual(pb(Tensor(ones: [3, 3])), Tensor(repeating: 5.9604645e-08, shape: [3, 3]))
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
        let expected: Tensor<Float> = Tensor([4, 8, 12])
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
        let expected: Tensor<Float> = Tensor([1, 2, 3])

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

    func testBatchNormalized() {
        let x = Tensor<Float>([
            [  -1.0474433,  -0.11914538,  -0.08634827,   0.15446888,    1.0572497],
            [   1.5165012,    0.3753972,  -0.30856386,   -0.3100725,   -1.9584457],
            [ 0.006384419,    1.4424847,   0.91568077,   0.66328526,   -1.0794537],
            [    1.056803,   0.14263044,   -1.8308276,    0.4189805,    0.6933893],
            [  0.30175626,  -0.16121633,   -0.4191958,  -0.53092813, -0.029484272]])
        let computedGradient = gradient(at: x) { $0.batchNormalized(alongAxis: 1).squared().sum() }
        // The expected value of the gradient was computed using the following Python code:
        // ```
        // import tensorflow as tf
        // with tf.GradientTape() as t:
        //   t.watch(x)
        //   mean, var = tf.nn.moments(x, axes=1, keepdims=True)
        //   y = tf.reduce_sum(tf.square(tf.nn.batch_normalization(
        //   x, mean, var, offset=0, scale=1, variance_epsilon=0.001)))
        // print(t.gradient(y, x))
        // ```
        let expectedGradient = Tensor<Float>([
            [-1.0127544e-02, -1.0807812e-03, -7.6115131e-04,  1.5857220e-03,  1.0383606e-02],
            [ 2.0323221e-03,  6.2976527e-04, -2.1077941e-04, -2.1265696e-04, -2.2384699e-03],
            [-1.3483668e-03,  3.7030075e-03,  1.8500184e-03,  9.6232636e-04, -5.1673558e-03],
            [ 1.8438101e-03,  8.9146197e-05, -3.6990643e-03,  6.1964989e-04,  1.1463165e-03],
            [ 1.2142579e-01,  1.7060755e-03, -6.5005139e-02, -9.3897656e-02,  3.5770576e-02]])
        assertEqual(computedGradient, expectedGradient, accuracy: 0.0001)
    }

    func testProductGrad() {
        // The expected gradient values were computed using the following Python code:
        // ```
        // import tensorflow as tf
        // # Adjust values of `x` and `axis` for each test.
        // x = tf.constant([[[3, 4], [5, 6], [7, 8]], [[3, 5], [0, 6], [5, 6]]], dtype=tf.float32)
        // axis = 1
        // with tf.GradientTape() as t:
        //   t.watch(x)
        //   y = tf.reduce_prod(x, axis=axis)
        //   z = tf.reduce_sum(y)
        // print(t.gradient(z, x))
        // ```
        func product(_ x: Tensor<Float>) -> Tensor<Float> {
            return x.product().sum()
        }
        func productSqueezingAxes1(_ x: Tensor<Float>) -> Tensor<Float> {
            return x.product(squeezingAxes: 1).sum()
        }
        func productSqueezingAxes_Neg1(_ x: Tensor<Float>) -> Tensor<Float> {
            return x.product(squeezingAxes: -1).sum()
        }
        func productSqueezingAxes01(_ x: Tensor<Float>) -> Tensor<Float> {
            return x.product(squeezingAxes: [0, 1]).sum()
        }
        XCTAssertEqual(gradient(at: [[10], [20]], in: product), [[20], [10]])
        XCTAssertEqual(gradient(at: [[10, 20], [20, 30]], in: productSqueezingAxes1),
                       [[20, 10], [30, 20]])
        XCTAssertEqual(gradient(at: [[10, 20], [20, 30]], in: productSqueezingAxes_Neg1),
                       [[20, 10], [30, 20]])
        XCTAssertEqual(gradient(at: [[[3, 4], [5, 6], [7, 8]], [[3, 5], [0, 6], [5, 6]]],
                                in: productSqueezingAxes1),
                       [[[35, 48], [21, 32], [15, 24]], [[0, 36], [15, 30], [0, 30]]])
        XCTAssertEqual(gradient(at: [[[3, 4], [5, 6], [7, 8]], [[3, 5], [0, 6], [5, 6]]],
                                in: productSqueezingAxes01),
                       [[[0, 8640], [0, 5760], [0, 4320]], [[0, 6912], [1575, 5760], [0, 5760]]])
    }

    static var allTests = [
        ("testSimpleGrad", testSimpleGrad),
        ("testGenericGrad", testGenericGrad),
        ("testConditionals", testConditionals),
        ("testNestedConditionals", testNestedConditionals),
        ("testRecursion", testRecursion),
        ("testScalarGenericGrad", testScalarGenericGrad),
        ("testScalarized", testScalarized),
        ("testScalars", testScalars),
        ("testInitFromScalars", testInitFromScalars),
        ("testInitFromScalarsWithShape", testInitFromScalarsWithShape),
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
        ("testMin", testMin),
        ("testMax", testMax),
        ("testTensorInitStacking", testTensorInitStacking),
        ("testExpandingShape", testExpandingShape),
        ("testSqueezingShape", testSqueezingShape),
        ("testReshapedBackprop", testReshapedBackprop),
        ("testReshaped", testReshaped),
        ("testConcatenationPlusPlus", testConcatenationPlusPlus),
        ("testConcatenated", testConcatenated),
        ("testTransposed", testTransposed),
        ("testSigmoid", testSigmoid),
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
        ("testUnbroadcastLike", testUnbroadcastLike),
        ("testBatchNormalized", testBatchNormalized),
        ("testProductGrad", testProductGrad),
    ]
}
