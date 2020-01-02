// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

/// Stochastic gradient descent (SGD) optimizer.
///
/// An optimizer that implements stochastic gradient descent, with support for momentum, learning
/// rate decay, and Nesterov momentum.
public class SGD<Model: Differentiable>: Optimizer
    where Model.TangentVector: VectorProtocol & ElementaryFunctions,
          Model.TangentVector.VectorSpaceScalar == Float {
    public typealias Model = Model
    /// The learning rate.
    public var learningRate: Float
    /// The momentum factor. It accelerates stochastic gradient descent in the relevant direction
    /// and dampens oscillations.
    public var momentum: Float
    /// The weight decay.
    public var decay: Float
    /// Use Nesterov momentum if true.
    public var nesterov: Bool
    /// The velocity state of the model.
    public var velocity: Model.TangentVector = .zero
    /// The set of steps taken.
    public var step: Int = 0

    public init(
        for model: __shared Model,
        learningRate: Float = 0.01,
        momentum: Float = 0,
        decay: Float = 0,
        nesterov: Bool = false
    ) {
        precondition(learningRate >= 0, "Learning rate must be non-negative")
        precondition(momentum >= 0, "Momentum must be non-negative")
        precondition(decay >= 0, "Weight decay must be non-negative")

        self.learningRate = learningRate
        self.momentum = momentum
        self.decay = decay
        self.nesterov = nesterov
    }

    public func update(_ model: inout Model, along direction: Model.TangentVector) {
        step += 1
        let learningRate = self.learningRate * 1 / (1 + decay * Float(step))
        velocity = momentum * velocity - direction * learningRate
        if nesterov {
            model.move(along: momentum * velocity - direction * learningRate)
        } else {
            model.move(along: velocity)
        }
    }
}
