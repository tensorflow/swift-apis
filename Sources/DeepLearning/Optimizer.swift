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

#if !COMPILING_TENSORFLOW_MODULE
import TensorFlow
#endif

/// A machine learning optimizer.
///
/// Optimizers apply an optimization algorithm to update the differentiable variables of a machine
/// learning model.
public protocol Optimizer {
    /// The type of the model whose parameters are optimized.
    associatedtype Model: Layer
    /// The scalar parameter type.
    associatedtype Scalar: FloatingPoint
    /// The learning rate.
    var learningRate: Scalar { get }
    /// Updates the specified differentiable variables along the specified
    /// direction.
    mutating func update(_ variables: inout Model.AllDifferentiableVariables,
                         along direction: Model.CotangentVector)
}

// MARK: - Key-path based optimizers

/// Adam optimizer.
///
/// - Reference: ["Adam - A Method for Stochastic Optimization"](
///   https://arxiv.org/abs/1412.6980v8)
public class Adam<Model: Layer, Scalar: TensorFlowFloatingPoint>: Optimizer
    where Model.AllDifferentiableVariables == Model.CotangentVector {
    /// The learning rate.
    public let learningRate: Scalar
    /// A coefficient used to calculate the first and second moments of
    /// gradients.
    public var beta1: Scalar
    /// A coefficient used to calculate the first and second moments of
    /// gradients.
    public var beta2: Scalar
    /// A small scalar added to the denominator to improve numerical stability.
    public let epsilon: Scalar
    /// The weight decay.
    public let decay: Scalar

    public init(
        learningRate: Scalar = 1e-3,
        beta1: Scalar = 0.9,
        beta2: Scalar = 0.999,
        epsilon: Scalar = 1e-8,
        decay: Scalar = 0
    ) {
        precondition(learningRate >= 0, "Learning rate must be non-negative")
        precondition(0 <= beta1 && beta1 <= 1, "Beta parameter must be between 0 and 1")
        precondition(0 <= beta2 && beta2 <= 1, "Beta parameter must be between 0 and 1")
        precondition(decay >= 0, "Weight decay must be non-negative")

        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay
    }

    public convenience init(
        for _: __shared Model,
        learningRate: Scalar = 1e-3,
        beta1: Scalar = 0.9,
        beta2: Scalar = 0.999,
        epsilon: Scalar = 1e-8,
        decay: Scalar = 0,
        scalarType: Scalar.Type
    ) {
        self.init(
            learningRate: learningRate,
            beta1: beta1,
            beta2: beta2,
            epsilon: epsilon,
            decay: decay)
    }

    private var step: Scalar = 0
    private var firstMoments = Model.AllDifferentiableVariables.zero
    private var secondMoments = Model.AllDifferentiableVariables.zero

    public func update(_ model: inout Model.AllDifferentiableVariables,
                       along direction: Model.AllDifferentiableVariables) {
        step += 1
        let learningRate = self.learningRate * 1 / (1 + decay * step)
        let stepSize = learningRate * (sqrt(1 - pow(beta2, step)) / (1 - pow(beta1, step)))
        for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Scalar>.self) {
            firstMoments[keyPath: kp] =
                firstMoments[keyPath: kp] * beta1 + (1 - beta1) * direction[keyPath: kp]
            secondMoments[keyPath: kp] =
                secondMoments[keyPath: kp] * beta2 + (1 - beta2) *
                     direction[keyPath: kp] * direction[keyPath: kp]
            model[keyPath: kp] -=
                stepSize * firstMoments[keyPath: kp] / (sqrt(secondMoments[keyPath: kp]) + epsilon)
        }
    }
}

/// RMSProp optimizer.
///
/// It is recommended to leave the parameters of this optimizer at their default values (except the
/// learning rate, which can be freely tuned). This optimizer is usually a good choice for recurrent
/// neural networks.
///
/// - Reference: ["rmsprop: Divide the gradient by a running average of its recent magnitude"](
///   http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
public class RMSProp<Model: Layer, Scalar: TensorFlowFloatingPoint>: Optimizer
    where Model.AllDifferentiableVariables == Model.CotangentVector {
    /// The learning rate.
    public let learningRate: Scalar
    // TODO: Document `rho`. Keras doesn't document `rho`.
    public let rho: Scalar
    /// A small scalar added to the denominator to improve numerical stability.
    public let epsilon: Scalar
    /// The weight decay.
    public let decay: Scalar

    public init(
        learningRate: Scalar = 0.001,
        rho: Scalar = 0.9,
        epsilon: Scalar = 1e-8,
        decay: Scalar = 0
    ) {
        precondition(learningRate >= 0, "Learning rate must be non-negative")
        precondition(rho >= 0, "Rho must be non-negative")
        precondition(decay >= 0, "Weight decay must be non-negative")

        self.learningRate = learningRate
        self.rho = rho
        self.epsilon = epsilon
        self.decay = decay
    }

    public convenience init(
        for _: __shared Model,
        learningRate: Scalar = 0.001,
        rho: Scalar = 0.9,
        epsilon: Scalar = 1e-8,
        decay: Scalar = 0,
        scalarType: Scalar.Type
    ) {
        self.init(learningRate: learningRate, rho: rho, epsilon: epsilon, decay: decay)
    }

    private var step: Scalar = 0
    private var alpha = Model.AllDifferentiableVariables.zero

    public func update(_ model: inout Model.AllDifferentiableVariables,
                       along direction: Model.CotangentVector) {
        step += 1
        let learningRate = self.learningRate * 1 / (1 + decay * step)
        for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Scalar>.self) {
            alpha[keyPath: kp] =
                rho * alpha[keyPath: kp] + (1 - rho) * pow(direction[keyPath: kp], 2)
            model[keyPath: kp] -=
                learningRate * direction[keyPath: kp] / (sqrt(alpha[keyPath: kp]) + epsilon)
        }
    }
}

/// Stochastic gradient descent (SGD) optimizer.
///
/// An optimizer that implements stochastic gradient descent, with support for momentum, learning
/// rate decay, and Nesterov momentum.
public class SGD<Model: Layer, Scalar: TensorFlowFloatingPoint>: Optimizer
    where Model.AllDifferentiableVariables == Model.CotangentVector {
    /// The learning rate.
    public let learningRate: Scalar
    /// The momentum factor. It accelerates stochastic gradient descent in the relevant direction
    /// and dampens oscillations.
    public let momentum: Scalar
    /// The weight decay.
    public let decay: Scalar
    /// Use Neseterov momentum if true.
    public let nesterov: Bool

    public init(
        learningRate: Scalar = 0.01,
        momentum: Scalar = 0,
        decay: Scalar = 0,
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

    public convenience init(
        for _: __shared Model,
        learningRate: Scalar = 0.01,
        momentum: Scalar = 0,
        decay: Scalar = 0,
        nesterov: Bool = false,
        scalarType: Scalar.Type
    ) {
        self.init(learningRate: learningRate, momentum: momentum, decay: decay, nesterov: nesterov)
    }

    private var step: Scalar = 0
    private var velocity = Model.AllDifferentiableVariables.zero

    public func update(_ model: inout Model.AllDifferentiableVariables,
                       along direction: Model.CotangentVector) {
        step += 1
        let learningRate = self.learningRate * 1 / (1 + decay * step)
        for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Scalar>.self) {
            velocity[keyPath: kp] =
                momentum * velocity[keyPath: kp] - learningRate * direction[keyPath: kp]
            if nesterov {
                model[keyPath: kp] +=
                    momentum * velocity[keyPath: kp] - learningRate * direction[keyPath: kp]
            } else {
                model[keyPath: kp] += velocity[keyPath: kp]
            }
        }
    }
}

// MARK: - Manifold optimizers

/// A Riemann manifold stochastic gradient descent (SGD) optimizer.
public class RiemannSGD<Model: Layer, Scalar: FloatingPoint>: Optimizer
    where Model.TangentVector: VectorNumeric, Model.TangentVector.Scalar == Scalar {
    /// The learning rate.
    public var learningRate: Scalar

    public init(learningRate: Scalar) {
        self.learningRate = learningRate
    }

    public convenience init(
        for _: __shared Model,
        learningRate: Scalar,
        scalarType _: Scalar.Type
    ) {
        self.init(learningRate: learningRate)
    }

    public func update(_ model: inout Model.AllDifferentiableVariables,
                       along direction: Model.CotangentVector) {
        model = model.moved(along: learningRate * (.zero - model.tangentVector(from: direction)))
    }
}
