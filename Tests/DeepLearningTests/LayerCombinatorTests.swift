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

final class LayerCombinatorTests: XCTestCase {
    func testSequential() {
        let inputSize = 2
        let hiddenSize = 4
        let model =
            Dense<Float>(inputSize: inputSize, outputSize: hiddenSize, activation: relu) >>
            Dense<Float>(inputSize: hiddenSize, outputSize: 1, activation: relu)
        
        let optimizer = SGD<model.type, Float>(learningRate: 0.02)  // Doesn't compile... :-(
        let x: Tensor<Float> = [[0, 0], [0, 1], [1, 0], [1, 1]]
        let y: Tensor<Float> = [0, 1, 1, 0]
        
        for _ in 0..<1000 {
            let (_, 𝛁model) = model.valueWithGradient { model -> Tensor<Float> in
                let ŷ = model.applied(to: x)
                return meanSquaredError(predicted: ŷ, expected: y)
            }
            optimizer.update(&model.allDifferentiableVariables, along: 𝛁model)
        }
        print(model.applied(to: [[0, 0], [0, 1], [1, 0], [1, 1]]))
    }
    
    static var allTests = [
        ("testSequential", testSequential)
    ]
}