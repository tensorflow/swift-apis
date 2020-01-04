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

final class LinearAlgebraTests: XCTestCase {
    func testCholesky() {
        let shapes = [[3, 3], [4, 2, 2], [2, 1, 16, 16]]
        let permutations = [[1, 0], [0, 2, 1], [0, 1, 3, 2]] // To avoid permuting batch dimensions.
        for (shape, permutation) in zip(shapes, permutations) {
            let a = Tensor<Float>(randomNormal: TensorShape(shape))
            let x = matmul(a, a.transposed(permutation: permutation)) // Make `a` positive-definite.
            let l = cholesky(x)
            let xReconstructed = matmul(l, l.transposed(permutation: permutation))
            assertEqual(xReconstructed, x, accuracy: 1e-5)
        }

        // The expected value of the gradient was computed using the following Python code:
        // ```
        // import tensorflow as tf
        // x = tf.constant([[[6., 4.], [4., 6.]], [[2., 6.], [6., 20.]]])
        // with tf.GradientTape() as tape:
        //    tape.watch(x)
        //    l = tf.reduce_sum(tf.linalg.cholesky(x))
        // print(tape.gradient(l, x))
        // ```
        let x = Tensor<Float>([[[6, 4], [4, 6]], [[2, 6], [6, 20]]])
        let computedGradient = gradient(at: x) { cholesky($0).sum() }
        let expectedGradient = Tensor<Float>([
            [[0.1897575, 0.02154995],
             [0.02154995, 0.2738613]],
            [[2.4748755, -0.7071073],
             [-0.7071073, 0.3535535]]])
        assertEqual(computedGradient, expectedGradient, accuracy: 1e-5) 
    }
    
    func testQRDecompositionApproximation() {
        let shapes = [[5, 8], [3, 4, 4], [3, 3, 32, 64]]
        for shape in shapes {
            let a = Tensor<Float>(randomNormal: TensorShape(shape))
            let (q, r) = a.qrDecomposition()
            let aReconstituted = matmul(q,r)
            assertEqual(aReconstituted, a, accuracy: 1e-5)

            let (qFull, rFull) = a.qrDecomposition(fullMatrices: true)
            let aReconstitutedFull = matmul(qFull, rFull)
            assertEqual(aReconstitutedFull, a, accuracy: 1e-5)
        }
    }
    
    func testTrace() {
        assertEqual(trace(Tensor<Float>(ones: [3, 3])), Tensor(3.0), accuracy: 1e-16)
        assertEqual(trace(Tensor<Float>(ones: [5, 6])), Tensor(5.0), accuracy: 1e-16)
        let shapes = [[1, 3, 3], [2, 4, 4], [2, 3, 5, 5]]
        for shape in shapes {
            let x = Tensor<Float>(ones: TensorShape(shape))
            let computedTrace = trace(x)
            let leadingShape = TensorShape(shape.dropLast(2))
            let value = Float(shape.last!)
            let expectedTrace = Tensor<Float>(repeating: value, shape: leadingShape)
            assertEqual(computedTrace, expectedTrace, accuracy: 1e-16)
        }
    }

    func testTraceGradient() {
        let shape: TensorShape = [2, 4, 4]
        let scalars = (0..<shape.contiguousSize).map(Float.init)
        let x = Tensor<Float>(shape: shape, scalars: scalars)
        let computedGradient = gradient(at: x) { (trace($0) * [2.0, 3.0]).sum() }
        let a = Tensor<Float>(repeating: 2.0, shape: [4]).diagonal()
        let b = Tensor<Float>(repeating: 3.0, shape: [4]).diagonal()
        let expectedGradient = Tensor([a, b])
        assertEqual(computedGradient, expectedGradient, accuracy: 1e-16)
    }

    func testLogdet() {
        let input = Tensor<Float>([[[6.0, 4.0], [4.0, 6.0]], [[2.0, 6.0], [6.0, 20.0]]])
        let expected = Tensor<Float>([2.9957323, 1.3862934])
        let computed = logdet(input)
        assertEqual(computed, expected, accuracy: 1e-5)
    }
    
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // x = tf.constant([[[6., 4.], [4., 6.]], [[2., 6.], [6., 20.]]])
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = tf.reduce_sum(tf.linalg.logdet(x))
    // print(tape.gradient(y, x))
    // ```
    func testLogdetGradient() {
        let input = Tensor<Float>([[[6.0, 4.0], [4.0, 6.0]], [[2.0, 6.0], [6.0, 20.0]]])
        let expectedGradient = Tensor<Float>([
            [[ 0.29999998, -0.2       ],
             [-0.2       ,  0.3       ]],
            [[ 5.0000043 , -1.5000012 ],
             [-1.5000012 ,  0.50000036]]])
        let computedGradient = gradient(at: input) { logdet($0).sum() }
        assertEqual(computedGradient, expectedGradient, accuracy: 1e-5)
    }
    
    
    /// Data generation function for the triangular solve test.
    /// Value of the gradient was computed using the following code:
    ///
    /// ```
    /// import tensorflow as tf
    /// a = tf.Variable([[1., 0., 0.],
    ///                  [2., 1., 0.],
    ///                  [3., 2., 1.]])
    /// b = tf.Variable([[1.], [1.], [3.]])
    /// with tf.GradientTape() as tape:
    ///     x = tf.reduce_sum(tf.linalg.triangular_solve(a, b))
    /// grad = tape.gradient(x, [a, b])
    /// ```
    func triangularSolveTestData(
    ) -> (
        a: Tensor<Float>,
        x: Tensor<Float>,
        b: Tensor<Float>,
        aGrad: Tensor<Float>,
        bGrad: Tensor<Float>,
        leadingShapes: [([Int], [Int])]
    ) {
        typealias IntList = [Int]
        typealias PairIntList = (IntList, IntList)

        let a = Tensor<Float>([
            [1, 0, 0],
            [2, 1, 0],
            [3, 2, 1]
        ])
        let b = Tensor<Float>([1,  1, 3]).reshaped(to: [-1, 1])
        let x = Tensor<Float>([1, -1, 2]).reshaped(to: [-1, 1])
        let aGrad = Tensor<Float>([
            [ 0,  0,  0],
            [ 1, -1,  0],
            [-1,  1, -2]
        ])
        let bGrad = Tensor<Float>([0, -1, 1]).reshaped(to: [-1, 1])
        let dimEmpty: IntList = []
        let dimOne: IntList = [2]
        let dimTwo: IntList = [3, 2]
        let dims: [IntList] = [dimEmpty, dimOne, dimTwo]
        let leadingShapes: [PairIntList] = Array(zip(dims, dims))
            + Array(zip([dimEmpty, dimEmpty, dimOne,   dimTwo],
                        [dimOne,   dimTwo,   dimEmpty, dimEmpty]))
            + Array(zip([dimOne, dimTwo],
                        [dimTwo, dimOne]))
        return (a, x, b, aGrad, bGrad, leadingShapes)
    }
    
    func testTriangularSolve() {
        let (a, x, b, aGrad, bGrad, leadingShapes) = triangularSolveTestData()
        for (aLeadingShape, bLeadingShape) in leadingShapes {
            let aNewShape = aLeadingShape + a.shape
            let bNewShape = bLeadingShape + b.shape
            let aNew = a.broadcasted(to: aNewShape)
            let bNew = b.broadcasted(to: bNewShape)
            let multiplier = Float(extractLeadingDims(aNew, bNew, ignoreLast: 2).contiguousSize)
            let xComputed = triangularSolve(matrix: aNew, rhs: bNew)
            let (aGradComputed, bGradComputed) = gradient(at: aNew, bNew) {
                triangularSolve(matrix: $0, rhs: $1).sum()
            }

            let xExpectedShape = (aNew.rank > bNew.rank ? aLeadingShape : bLeadingShape) + x.shape
            let xExpected = x.broadcasted(to: xExpectedShape)
            let aGradExpected = (aNew.rank > bNew.rank ? aGrad : multiplier * aGrad).broadcasted(like: aNew)
            let bGradExpected = (bNew.rank > aNew.rank ? bGrad : multiplier * bGrad).broadcasted(like: bNew)

            assertEqual(xComputed, xExpected, accuracy: 1e-16)
            assertEqual(aGradComputed, aGradExpected, accuracy: 1e-16)
            assertEqual(bGradComputed, bGradExpected, accuracy: 1e-16)
        }
    }
    
    func testExtractLeadingDims() {
        var a: TensorShape = []
        var b: TensorShape = []
        var computed1: [Int] = extractLeadingDims(a, b).dimensions
        var computed2: [Int] = extractLeadingDims(b, a).dimensions
        XCTAssertEqual(computed1, [Int]())
        XCTAssertEqual(computed2, [Int]())
        
        a = [1]
        b = [3, 2, 1]
        computed1 = extractLeadingDims(a, b).dimensions
        computed2 = extractLeadingDims(b, a).dimensions
        XCTAssertEqual(computed1, [3, 2])
        XCTAssertEqual(computed2, [3, 2])

        a = [3, 2, 1]
        b = [4, 5, 6, 3, 20, 10]
        computed1 = extractLeadingDims(a, b, ignoreLast: 2).dimensions
        computed2 = extractLeadingDims(b, a, ignoreLast: 2).dimensions
        XCTAssertEqual(computed1, [4, 5, 6])
        XCTAssertEqual(computed2, [4, 5, 6])

        a = [3, 2, 1]
        b = [30, 20, 10]
        computed1 = extractLeadingDims(a, b, ignoreLast: 3).dimensions
        computed2 = extractLeadingDims(b, a, ignoreLast: 3).dimensions
        XCTAssertEqual(computed1, [Int]())
        XCTAssertEqual(computed2, [Int]())
    }
    
    static var allTests = [
        ("testCholesky", testCholesky),
        ("testQRDecompositionApproximation", testQRDecompositionApproximation),
        ("testTrace", testTrace),
        ("testTraceGradient", testTraceGradient),
        ("testLogdet", testLogdet),
        ("testLogdetGradient", testLogdetGradient),
        ("testTriangularSolve", testTriangularSolve),
        ("testExtractLeadingDims", testExtractLeadingDims)
    ]
}
