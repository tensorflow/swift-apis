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
        struct Classifier: Layer {
            private static var rng = ARC4RandomNumberGenerator(seed: 42)
            var l1 = Dense<Float>(inputSize: 2, outputSize: 4)
            var l2 = Dense<Float>(inputSize: 4, outputSize: 1)
            func applied(to input: Tensor<Float>) -> Tensor<Float> {
                let h1 = l1.applied(to: input)
                return l2.applied(to: h1)
            }
        }
        let optimizer = SGD<Classifier, Float>()
        var classifier = Classifier()
        for _ in 0..<10 {
            optimizer.update(&classifier.allDifferentiableVariables, along: .zero)
        }
    }

    static var allTests = [
        ("testXOR", testXOR),
    ]
}
