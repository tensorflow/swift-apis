// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
@testable import DeepLearning

final class TrivialModelTests: XCTestCase {
    func testXOR() {
        // NOTE: Blocked by https://bugs.swift.org/browse/SR-9656.
        struct Classifier : Layer {
            private static var rng = ARC4RandomNumberGenerator(seed: 42)
            var w1 = Tensor<Float>(randomUniform: [2, 4], generator: &rng)
            var w2 = Tensor<Float>(randomUniform: [4, 1], generator: &rng)
            var b1 = Tensor<Float>(zeros: [1, 4])
            var b2 = Tensor<Float>(zeros: [1, 1])
            func applied(to input: Tensor<Float>) -> Tensor<Float> {
                let o1 = sigmoid(matmul(input, w1) + b1)
                return sigmoid(matmul(o1, w2) + b2)
            }
        }
    }

    static var allTests = [
        ("testXOR", testXOR),
    ]
}
