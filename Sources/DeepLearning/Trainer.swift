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

/// SupervisedLearningTrainer implements a standard training loop for a supervised learning model.
///
/// Use the fit function to optimize a model with the provided optimizer and loss function against
/// the input features `input` attempting to predict the `output` values.
///
/// Note: the current implementation lacks important features, and is currently most useful as a
/// starting point to develop your own training loops.
public struct SupervisedLearningTrainer {
    private init() {}  // Users should not instantiate SupervisedLearningTrainer

    static func fit<Opt: Optimizer, LossScalar: TensorFlowFloatingPoint>(
        model: Opt.Model,
        parameters: inout Opt.Model.AllDifferentiableVariables,
        using optimizer: Opt,
        loss lossFn: @escaping @differentiable (Opt.Model.Output, Opt.Model.Output) -> Tensor<LossScalar>,
        input x: Opt.Model.Input,
        output y: Opt.Model.Output,
        stepCount: Int)
        where Opt.Model.AllDifferentiableVariables == Opt.Model.CotangentVector {
        // TODO: Rewrite training loop to be more flexible by using callbacks.
        let context = Context(learningPhase: .training)

        // TODO: Implement shuffling, randomization, etc!
        for _ in 0..<stepCount {
            let (𝛁model, _) = model.gradient(at: y) { model, y -> Tensor<LossScalar> in
                let ŷ = model.applied(to: x, in: context)
                return lossFn(ŷ, y)
            }
            // TODO: Accumulate loss & print out status updates.
            optimizer.update(&parameters, along: 𝛁model)
        }
    }
}
