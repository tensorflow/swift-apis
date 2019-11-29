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
    
    static var allTests = [("testCholesky", testCholesky)]
}
