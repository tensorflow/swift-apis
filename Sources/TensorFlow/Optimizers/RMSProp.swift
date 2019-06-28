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

/// RMSProp optimizer.
///
/// It is recommended to leave the parameters of this optimizer at their default values (except for 
/// the learning rate, which can be freely tuned). This optimizer is usually a good choice for 
/// recurrent neural networks.
///
/// Reference: ["rmsprop: Divide the gradient by a running average of its recent magnitude"](
/// http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
public class RMSProp<Model: Differentiable>: Optimizer
    where Model.TangentVector: VectorProtocol & PointwiseMultiplicative & ElementaryFunctions,
          Model.TangentVector.VectorSpaceScalar == Float {
    public typealias Model = Model
    /// The learning rate.
    public var learningRate: Float
    // TODO: Document `rho`. Keras doesn't document `rho`.
    public var rho: Float
    /// A small scalar added to the denominator to improve numerical stability.
    public var epsilon: Float
    /// The weight decay.
    public var decay: Float
    /// The step count.
    public var step: Float = 0
    /// The alpha values for all model differentiable variables.
    public var alpha: Model.TangentVector = .zero

    public init(
        for model: __shared Model,
        learningRate: Float = 0.001,
        rho: Float = 0.9,
        epsilon: Float = 1e-8,
        decay: Float = 0
    ) {
        precondition(learningRate >= 0, "Learning rate must be non-negative")
        precondition(rho >= 0, "Rho must be non-negative")
        precondition(decay >= 0, "Weight decay must be non-negative")

        self.learningRate = learningRate
        self.rho = rho
        self.epsilon = epsilon
        self.decay = decay
    }

    public func update(_ model: inout Model, along direction: Model.TangentVector) {
        update(&model.allDifferentiableVariables, along: direction)
    }

    // TODO: Deprecate this when `Differentiable.AllDifferentiableVariables` is removed.
    public func update(
        _ model: inout Model.AllDifferentiableVariables,
        along direction: Model.TangentVector
    ) {
        step += 1
        let learningRate = self.learningRate * 1 / (1 + decay * Float(step))
        alpha = alpha * rho + direction .* direction * (1 - rho)
        let denominator = Model.TangentVector.sqrt(alpha) + epsilon
        model.move(along: -learningRate * direction ./ denominator)
    }
}
