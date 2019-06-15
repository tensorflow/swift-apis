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

/// A machine learning optimizer.
///
/// Optimizers apply an optimization algorithm to update the differentiable variables of a machine
/// learning model.
public protocol Optimizer {
    /// The type of the model whose parameters are optimized.
    associatedtype Model: Differentiable
    /// The scalar parameter type.
    associatedtype Scalar: FloatingPoint
    /// The learning rate.
    var learningRate: Scalar { get set }
    /// Updates the specified differentiable variables along the specified
    /// direction.
    mutating func update(_ variables: inout Model, along direction: Model.TangentVector)
}

fileprivate extension Tensor where Scalar: Numeric {
    mutating func resetToZero() {
        self = Tensor(zeros: shape)
    }
}

/// Adam optimizer.
///
/// Reference: ["Adam - A Method for Stochastic Optimization"](
/// https://arxiv.org/abs/1412.6980v8)
public class Adam<Model: Differentiable>: Optimizer where Model.TangentVector: VectorProtocol {
    /// The learning rate.
    public var learningRate: Float
    /// A coefficient used to calculate the first and second moments of
    /// gradients.
    public var beta1: Float
    /// A coefficient used to calculate the first and second moments of
    /// gradients.
    public var beta2: Float
    /// A small scalar added to the denominator to improve numerical stability.
    public var epsilon: Float
    /// The learning rate decay.
    public var decay: Float
    /// The current step.
    public var step: Int = 0
    /// The first moments of the weights.
    public var firstMoments: Model.TangentVector = .zero
    /// The second moments of the weights.
    public var secondMoments: Model.TangentVector = .zero

    public init(
        for model: __shared Model,
        learningRate: Float = 1e-3,
        beta1: Float = 0.9,
        beta2: Float = 0.999,
        epsilon: Float = 1e-8,
        decay: Float = 0
    ) {
        precondition(learningRate >= 0, "Learning rate must be non-negative")
        precondition(0 <= beta1 && beta1 <= 1, "Beta parameter must be between 0 and 1")
        precondition(0 <= beta2 && beta2 <= 1, "Beta parameter must be between 0 and 1")
        precondition(decay >= 0, "Learning rate decay must be non-negative")

        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay
    }


    public func update(_ model: inout Model, along direction: Model.TangentVector) {
        step += 1
        let learningRate = self.learningRate / (1 + decay * Float(step))
        // Note: `stepSize` is split into two lines to avoid the "compiler is unable to type-check
        // this expression in reasonable time" error.
        var stepSize = learningRate * sqrt(1 - pow(beta2, Float(step)))
        stepSize = stepSize / (1 - pow(beta1, Float(step)))
        // Update Float & Double Tensor variables.
        firstMoments = firstMoments * beta1 + (1 - beta1) * direction
        secondMoments = secondMoments * beta2 + (1 - beta2) * direction * direction
        model -= stepSize * firstMoments / (sqrt(secondMoments) + epsilon)
    }
}

/// RMSProp optimizer.
///
/// It is recommended to leave the parameters of this optimizer at their default values (except the
/// learning rate, which can be freely tuned). This optimizer is usually a good choice for recurrent
/// neural networks.
///
/// Reference: ["rmsprop: Divide the gradient by a running average of its recent magnitude"](
/// http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
public class RMSProp<Model: Differentiable>: Optimizer where Model.TangentVector: VectorProtocol {
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
        step += 1
        let learningRate = self.learningRate / (1 + decay * Float(step))
        alpha = rho * alpha + (1 - rho) * pow(direction, 2)
        model -= learningRate * direction / (sqrt(alpha) + epsilon)
    }
}

/// Stochastic gradient descent (SGD) optimizer.
///
/// An optimizer that implements stochastic gradient descent, with support for momentum, learning
/// rate decay, and Nesterov momentum.
public class SGD<Model: Differentiable>: Optimizer where Model.TangentVector: VectorProtocol {
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
            model += momentum * velocity - direction * learningRate
        } else {
            model += velocity
        }
    }
}

// MARK: - Manifold optimizers

/// A Riemann manifold stochastic gradient descent (SGD) optimizer.
public class RiemannSGD<Model: Differentiable>: Optimizer
    where Model.TangentVector: VectorProtocol {
    /// The learning rate.
    public var learningRate: Model.TangentVector.Scalar

    public init(learningRate: Model.TangentVector.Scalar) {
        self.learningRate = learningRate
    }

    public convenience init(
        for _: __shared Model,
        learningRate: Scalar
    ) {
        self.init(learningRate: learningRate)
    }

    public func update(_ model: inout Model, along direction: Model.TangentVector) {
        model.move(along: -learningRate * direction)
    }
}

/// AdaGrad optimizer.
///
/// Individually adapts the learning rates of all model parameters by scaling them inversely proportional to
/// the square root of the sum of all the historical squared values of the gradient.
///
/// Reference: ["Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"](
///   http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
///
public class AdaGrad<Model: Differentiable>: Optimizer where Model.TangentVector: VectorProtocol {
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
        alpha = rho + direction.squared()
        model.move(along: -learningRate * direction / (sqrt(alpha + epsilon)))
    }
}
