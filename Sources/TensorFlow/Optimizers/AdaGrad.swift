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

/// AdaGrad optimizer.
///
/// Individually adapts the learning rates of all model parameters by scaling them inversely 
/// proportional to the square root of the sum of all the historical squared values of the gradient.
///
/// Reference: ["Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"](
/// http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
public class AdaGrad<Model: Differentiable>: Optimizer
    where Model.TangentVector: VectorProtocol & PointwiseMultiplicative & ElementaryFunctions,
          Model.TangentVector.VectorSpaceScalar == Float {
    public typealias Model = Model
    /// The learning rate.
    public var learningRate: Float
    /// The smoothing factor (ρ). Typical values are `0.5`, `0.9`, and `0.99`, for smoothing over 2,
    /// 10, and 100 examples, respectively.
    public var rho: Float
    /// A small scalar added to the denominator to improve numerical stability.
    public var epsilon: Float
    /// The alpha values for all model differentiable variables.
    public var alpha: Model.TangentVector = .zero

    public init(
        for model: __shared Model,
        learningRate: Float = 0.001,
        rho: Float = 0.9,
        epsilon: Float = 1e-8
    ) {
        precondition(learningRate >= 0, "Learning rate must be non-negative")
        precondition(rho >= 0, "Rho must be non-negative")

        self.learningRate = learningRate
        self.rho = rho
        self.epsilon = epsilon
    }

    public func update(_ model: inout Model, along direction: Model.TangentVector) {
        update(&model.allDifferentiableVariables, along: direction)
    }

    // TODO: Deprecate this when `Differentiable.AllDifferentiableVariables` is removed.
    public func update(
        _ model: inout Model.AllDifferentiableVariables,
        along direction: Model.TangentVector
    ) {
        alpha = rho + direction .* direction
        let denominator = Model.TangentVector.sqrt(alpha) + epsilon
        model.move(along: -learningRate * direction ./ denominator)
    }
}
