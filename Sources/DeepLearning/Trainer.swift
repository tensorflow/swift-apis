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

/// SupervisedTrainer implements a standard training loop for a model.
public struct SupervisedTrainer {
    static func fit<Opt: Optimizer, LossScalar: TensorFlowFloatingPoint>(
        model: inout Opt.Model,
        with optimizer: Opt,
        loss lossFn: @escaping @differentiable (Opt.Model.Output, Opt.Model.Output) -> LossScalar,
        x: Opt.Model.Input,
        y: Opt.Model.Output,
        stepCount numSteps: Int)
        where Opt.Model.AllDifferentiableVariables == Opt.Model.CotangentVector {
        // TODO: Rewrite training loop with callbacks.
        let context = Context(learningPhase: .training)

        // TODO: Implement shuffling, randomization, etc!
        for _ in 0..<numSteps {
            let (𝛁model, _) = model.gradient(at: y) { model, y -> LossScalar in
                let ŷ = model.applied(to: x, in: context)
                return lossFn(ŷ, y)
            }
            // TODO: Accumulate loss & print out status updates.
            optimizer.update(&model.allDifferentiableVariables, along: 𝛁model)
        }
    }
}
