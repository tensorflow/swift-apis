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

/// ADADELTA optimizer.
///
/// ADADELTA is a more robust extension of AdaGrad. ADADELTA adapts learning rates based on a moving
/// window of gradient updates rather than by accumulating all past gradient norms. It can thus 
/// adapt faster to changing dynamics of the optimization problem space.
/// 
/// Reference: ["ADADELTA: An Adaptive Learning Rate Method"](https://arxiv.org/abs/1212.5701)
public class AdaDelta<Model: Differentiable>: Optimizer
    where Model.TangentVector: VectorProtocol & PointwiseMultiplicative & ElementaryFunctions,
          Model.TangentVector.VectorSpaceScalar == Float {
    public typealias Model = Model
    /// The learning rate.
    public var learningRate: Float
    /// The decay factor, corresponding to fraction of gradient to keep at each time step.
    public var rho: Float
    /// A small scalar added to the denominator to improve numerical stability.
    public var epsilon: Float
    /// The learning rate decay.
    public var decay: Float
    /// The current step.
    public var step: Int = 0
    /// The accumulated, exponentially decaying average of squared gradients.
    public var averageSquared: Model.TangentVector = .zero
    /// The accumulated parameter updates.
    public var accumulatedDelta: Model.TangentVector = .zero

    public init(
        for model: __shared Model,
        learningRate: Float = 1,
        rho: Float = 0.95,
        epsilon: Float = 1e-6,
        decay: Float = 0
    ) {
        precondition(learningRate >= 0, "Learning rate must be non-negative")
        precondition(0 <= rho && rho <= 1, "Rho parameter must be between 0 and 1")
        precondition(0 <= epsilon, "Epsilon parameter must be non-negative")
        precondition(decay >= 0, "Learning rate decay must be non-negative")

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
        let learningRate = self.learningRate / (1 + decay * Float(step))
        averageSquared = rho * averageSquared + (1 - rho) * direction .* direction
        var stepSize = direction .* Model.TangentVector.sqrt(accumulatedDelta + epsilon)
        stepSize ./= Model.TangentVector.sqrt(averageSquared + epsilon)
        model.move(along: -learningRate * stepSize)
        accumulatedDelta = rho * accumulatedDelta + (1 - rho) * stepSize .* stepSize
    }
}
