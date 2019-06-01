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

final class MathOperatorTests: XCTestCase {
    func testLog1p() {
        let x = Tensor<Float>([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        let y = log1p(x)
        assertEqual(y, log(1 + x), accuracy: 0.0001)
    }

    func testExpm1() {
        let x = Tensor<Float>([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        let y = expm1(x)
        assertEqual(y, exp(x - 1), accuracy: 0.0001)
    }

    func testSign() {
        let x = Tensor<Float>([[1, 2, -3, 4, 5], [1, 2, 3, 4, -5]])
        let y = sign(x)
        XCTAssertEqual(y, Tensor<Float>([[1, 1, -1, 1, 1], [1, 1, 1, 1, -1]]))
    }

    func testReduction() {
        // 2 x 5
        let x = Tensor<Float>([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        XCTAssertEqual(Tensor(30), x.sum())
        XCTAssertEqual(
            Tensor(shape: [5], scalars: [2, 4, 6, 8, 10]),
            x.sum(squeezingAxes: 0))
        XCTAssertEqual(
            Tensor(shape: [1, 5], scalars: [2, 4, 6, 8, 10]),
            x.sum(alongAxes: 0))

        XCTAssertEqual(Tensor(14400), x.product())
        XCTAssertEqual(
            Tensor(shape: [5], scalars: [1, 4, 9, 16, 25]),
            x.product(squeezingAxes: 0))
        XCTAssertEqual(
            Tensor(shape: [1, 5], scalars: [1, 4, 9, 16, 25]),
            x.product(alongAxes: 0))

        XCTAssertEqual(Tensor(3), x.mean())
        XCTAssertEqual(
            Tensor(shape: [5], scalars: [1, 2, 3, 4, 5]),
            x.mean(squeezingAxes: 0))
        XCTAssertEqual(
            Tensor(shape: [5], scalars: [1, 2, 3, 4, 5]),
            x.mean(alongAxes: 0))
        XCTAssertEqual(
            Tensor(shape: [2], scalars: [3, 3]),
            x.mean(squeezingAxes: 1))
        XCTAssertEqual(
            Tensor(shape: [1, 2], scalars: [3, 3]),
            x.mean(alongAxes: 1))

        XCTAssertEqual(Tensor(2), x.variance())
        XCTAssertEqual(
            Tensor(shape: [5], scalars: [0, 0, 0, 0, 0]),
            x.variance(squeezingAxes: 0))
        XCTAssertEqual(
            Tensor(shape: [5], scalars: [0, 0, 0, 0, 0]),
            x.variance(alongAxes: 0))
        XCTAssertEqual(
            Tensor(shape: [2], scalars: [2, 2]),
            x.variance(squeezingAxes: 1))
        XCTAssertEqual(
            Tensor(shape: [1, 2], scalars: [2, 2]),
            x.variance(alongAxes: 1))
    }

    func testArgmax() {
        // 2 x 3
        let x = Tensor<Float>([[0, 1, 2], [3, 4, 5]])
        let argmax0 = x.argmax(squeezingAxis: 0)
        let argmax1 = x.argmax(squeezingAxis: 1)
        let scalarsArgmax = x.argmax()
        XCTAssertEqual(ShapedArray(shape: [3], scalars: [1, 1, 1]), argmax0.array)
        XCTAssertEqual(ShapedArray(shape: [2], scalars: [2, 2]), argmax1.array)
        XCTAssertEqual(ShapedArray(shape: [], scalars: [5]), scalarsArgmax.array)
    }

    func testCeilAndFloor() {
        let x = Tensor<Float>([-1.3, -0.4, 0.5, 1.6])
        let xFloor = floor(x)
        let xCeil = ceil(x)
        XCTAssertEqual(ShapedArray(shape: [4], scalars: [-2, -1, 0, 1]), xFloor.array)
        XCTAssertEqual(ShapedArray(shape: [4], scalars: [-1, 0, 1, 2]), xCeil.array)
    }

    func testSimpleMath() {
        let x = Tensor<Float>([1.2, 1.2])
        let y = tanh(x)
        let array = y.array
        XCTAssertEqual([2], array.shape)
        XCTAssertEqual(0.833655, Double(array.scalars[0]), accuracy: 0.0001)
        XCTAssertEqual(0.833655, Double(array.scalars[1]), accuracy: 0.0001)
    }

    func testStandardDeviation() {
        XCTAssertEqual(Tensor(0), Tensor<Float>([1]).standardDeviation())
        XCTAssertEqual(Tensor(0.5), Tensor<Float>([0, 1]).standardDeviation(alongAxes: 0))
        XCTAssertEqual(Tensor(0.5), Tensor<Float>([0, 1]).standardDeviation())
        XCTAssertEqual(
            2.87228132,
            Tensor<Float>(rangeFrom: 0, to: 10, stride: 1).standardDeviation().scalarized(),
            accuracy: 0.001)
        let matrix = Tensor<Float>(rangeFrom: 0, to: 10, stride: 1).reshaped(to: [2, 5])
        XCTAssertEqual(2.87228132, matrix.standardDeviation().scalarized(), accuracy: 0.001)
        let values = matrix.standardDeviation(alongAxes: 1).array.scalars
        XCTAssertEqual(1.4142, Double(values[0]), accuracy: 0.0001)
        XCTAssertEqual(1.4142, Double(values[1]), accuracy: 0.0001)
    }

    func test3Adds() {
        let a = Tensor<Float>([1])
        let b = Tensor<Float>([2])
        let c = Tensor<Float>([3])

        let o = a + b + c
        XCTAssertEqual([6], o.scalars)
    }

    func testMultiOpMath() {
        let x = Tensor<Float>([1.2, 1.2])
        let y = Tensor<Float>([2.4, 2.4])
        let t1 = x + y
        let t2 = t1 * t1
        let t3 = sqrt(t2)

        let array1 = t1.array
        let array2 = t2.array
        let array3 = t3.array
        XCTAssertEqual([2], array1.shape)
        XCTAssertEqual([2], array2.shape)
        XCTAssertEqual([2], array3.shape)
        XCTAssertEqual(3.6, Double(array1.scalars[0]), accuracy: 0.0001)
        XCTAssertEqual(3.6, Double(array1.scalars[1]), accuracy: 0.0001)
        XCTAssertEqual(12.96, Double(array2.scalars[0]), accuracy: 0.0001)
        XCTAssertEqual(12.96, Double(array2.scalars[1]), accuracy: 0.0001)
        XCTAssertEqual(3.6, Double(array3.scalars[0]), accuracy: 0.0001)
        XCTAssertEqual(3.6, Double(array3.scalars[1]), accuracy: 0.0001)
    }

    func testXWPlusB() {
        // Shape: 1 x 4
        let x = Tensor<Float>([[1.0, 2.0, 2.0, 1.0]])
        // Shape: 4 x 2
        let w = Tensor<Float>([[1.0, 0.0], [3.0, 0.0], [2.0, 3.0], [1.0, 0.0]])
        // Shape: 2
        let b = Tensor<Float>([0.5, 0.5])
        // Shape: 1 x 2 (broadcasted)
        let result = matmul(x, w) + b
        XCTAssertEqual([1, 2], result.shape)
        XCTAssertEqual([12.5, 6.5], result.scalars)
    }

    func testXORInference() {
        func xor(_ x: Float, _ y: Float) -> Float {
            let x = Tensor<Float>([x, y]).reshaped(to: [1, 2])

            // FIXME: If params are declared outside of `xor`, it would crash.
            // 2 x 4
            let w1 = Tensor<Float>(
                [[-1.83586664, -0.20809225, 0.47667537, 1.90780607],
                [-1.83523219, -0.51167348, 0.15490439, 1.91018065]])
            // 1 x 4
            let b1 = Tensor<Float>([[2.54353216, 0.25132703, -0.16503136, -0.85754058]])
            // 4 x 1
            let w2 = Tensor<Float>([[3.04350065], [0.35590511], [-0.3252157], [3.49349223]])
            // 1 x 1
            let b2 = Tensor<Float>([[-0.74635993]])

            let o1 = tanh(matmul(x, w1) + b1)
            let y = tanh(matmul(o1, w2) + b2)
            return y.array.scalars[0] // TODO: use better scalar getter
        }

        XCTAssertEqual(0.0, xor(0.0, 0.0), accuracy: 0.1)
        XCTAssertEqual(1.0, xor(0.0, 1.0), accuracy: 0.1)
        XCTAssertEqual(1.0, xor(1.0, 0.0), accuracy: 0.1)
        XCTAssertEqual(0.0, xor(1.0, 1.0), accuracy: 0.1)
    }

    func testMLPClassifierStruct() {
        struct MLPClassifier {
            // 2 x 4
            var w1 = Tensor<Float>([[1.0, 0.8, 0.4, 0.4],
                                    [0.4, 0.3, 0.2, 0.1]])
            // 4 x 1
            var w2 = Tensor<Float>([[0.4], [0.4], [0.3], [0.9]])
            var b1 = Tensor<Float>(zeros: [1, 4])
            var b2 = Tensor<Float>(zeros: [1, 1])

            func prediction(for x: Tensor<Float>) -> Tensor<Float> {
                let o1 = tanh(matmul(x, w1) + b1)
                return tanh(matmul(o1, w2) + b2)
            }
        }

        let input = Tensor<Float>([[1, 0.5]])
        let classifier = MLPClassifier()
        let prediction = classifier.prediction(for: input)
        XCTAssertEqual(0.816997, Double(prediction.scalars[0]), accuracy: 0.0001)
    }

    func testBroadcastedAddGradient() {
	  func foo(_ x: Tensor<Float>, _ y: Tensor<Float>) -> Tensor<Float> {
	    return (x + y).sum()
	  }
	  let x = Tensor<Float>(ones: [1, 2, 1, 4])
	  let y = Tensor<Float>(ones: [4, 1, 3, 1])
	  let (dx, dy) = gradient(at: x, y, in: foo)
	  XCTAssertEqual(x.shape, dx.shape)
	  XCTAssertEqual(y.shape, dy.shape)
	}

    static var allTests = [
        ("testLog1p", testLog1p),
        // TODO(https://bugs.swift.org/browse/TF-543): Test is failing in Linux
	// ("testExpm1", testExpm1),
        ("testSign", testSign),
        ("testReduction", testReduction),
        ("testArgmax", testArgmax),
        ("testCeilAndFloor", testCeilAndFloor),
        ("testSimpleMath", testSimpleMath),
        ("testStandardDeviation", testStandardDeviation),
        ("test3Adds", test3Adds),
        ("testMultiOpMath", testMultiOpMath),
        ("testXWPlusB", testXWPlusB),
        ("testXORInference", testXORInference),
        ("testMLPClassifierStruct", testMLPClassifierStruct),
        ("testBroadcastedAddGradient", testBroadcastedAddGradient)
    ]
}
