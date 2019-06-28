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

/// A numerical optimizer.
///
/// Optimizers apply an optimization algorithm to update the differentiable models.
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
public class Adam<Model: Differentiable>: Optimizer
    where Model.TangentVector: VectorProtocol & PointwiseMultiplicative & ElementaryFunctions,
          Model.TangentVector.VectorSpaceScalar == Float,
          Model.AllDifferentiableVariables == Model.TangentVector {
    public typealias Model = Model
    /// The learning rate.
    public var learningRate: Float
    /// A coefficient used to calculate the first and second moments of the gradients.
    public var beta1: Float
    /// A coefficient used to calculate the first and second moments of the gradients.
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
        let learningRate = self.learningRate * 1 / (1 + decay * Float(step))
        // Note: `stepSize` and `secondMoments` are split into two lines to avoid the "compiler is 
        // unable to type-check this expression in reasonable time" error.
        var stepSize = learningRate * sqrt(1 - pow(beta2, Float(step)))
        stepSize = stepSize / (1 - pow(beta1, Float(step)))
        firstMoments = firstMoments * beta1 + direction * (1 - beta1)
        secondMoments = secondMoments * beta2
        secondMoments += direction .* direction * (1 - beta2)
        let denominator = Model.TangentVector.sqrt(secondMoments) + epsilon
        // TODO: Update this when `./` becomes available.
        model.move(along: -stepSize * firstMoments .* denominator.reciprocal)
    }

    // TODO: Deprecate this when `Differentiable.AllDifferentiableVariables` is removed.
    public func update(
        _ model: inout Model.AllDifferentiableVariables,
        along direction: Model.AllDifferentiableVariables
    ) {
        step += 1
        let learningRate = self.learningRate * 1 / (1 + decay * Float(step))
        // Note: `stepSize` and `secondMoments` are split into two lines to avoid the "compiler is 
        // unable to type-check this expression in reasonable time" error.
        var stepSize = learningRate * sqrt(1 - pow(beta2, Float(step)))
        stepSize = stepSize / (1 - pow(beta1, Float(step)))
        firstMoments = firstMoments * beta1 + direction * (1 - beta1)
        secondMoments = secondMoments * beta2
        secondMoments += direction .* direction * (1 - beta2)
        let denominator = Model.TangentVector.sqrt(secondMoments) + epsilon
        // TODO: Update this when `./` becomes available.
        model -= stepSize * firstMoments .* denominator.reciprocal
    }
}

/// AdaMax optimizer.
///
/// A variant of Adam based on the infinity-norm.
///
/// Reference: Section 7 of ["Adam - A Method for Stochastic Optimization"](
/// https://arxiv.org/abs/1412.6980v8)
public class AdaMax<Model: Layer>: Optimizer
    where Model.AllDifferentiableVariables == Model.TangentVector {
    public typealias Model = Model
    /// The learning rate.
    public var learningRate: Float
    /// Decay rate used to estimate the first moment (mean) of gradients.
    public var beta1: Float
    /// Decay rate used to estimate the exponentially weighted infinity norm.
    public var beta2: Float
    /// A small scalar added to the denominator to improve numerical stability.
    public var epsilon: Float
    /// The learning rate decay.
    public var decay: Float
    /// The step count.
    public var step: Int = 0
    /// The first moments of the weights.
    public var firstMoments: Model.TangentVector
    /// The exponentially weighted infinity norm of the weights.
    public var infinityNorm: Model.TangentVector

    /// Note: The default parameters follow those provided in the paper.
    public init(
        for model: __shared Model,
        learningRate: Float = 0.002,
        beta1: Float = 0.9,
        beta2: Float = 0.999,
        epsilon: Float = 1e-8,
        decay: Float = 0
    ) {
        precondition(learningRate >= 0, "Learning rate must be non-negative.")
        precondition(0 <= beta1 && beta1 <= 1, "Beta parameter must be between 0 and 1.")
        precondition(0 <= beta2 && beta2 <= 1, "Beta parameter must be between 0 and 1.")
        precondition(decay >= 0, "Learning rate decay must be non-negative.")

        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay

        // Initialize first moments and infinity norm to be zeros of the same shape.
        // We can't use `Model.AllDifferentiableVariables.zero` due to the
        // interaction between Key Paths and Differentiable Arrays.
        firstMoments = model.allDifferentiableVariables
        infinityNorm = model.allDifferentiableVariables
        for kp in firstMoments.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            firstMoments[keyPath: kp].resetToZero()
            infinityNorm[keyPath: kp].resetToZero()
        }
        for kp in firstMoments.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
            firstMoments[keyPath: kp].resetToZero()
            infinityNorm[keyPath: kp].resetToZero()
        }
    }

    // TODO: Deprecate this when `Differentiable.AllDifferentiableVariables` is removed.
    public func update(_ model: inout Model.AllDifferentiableVariables,
                       along direction: Model.AllDifferentiableVariables) {
        step += 1
        let learningRate = self.learningRate * 1 / (1 + decay * Float(step))
        // Note: `stepSize` is split into two lines to avoid the "compiler is unable to type-check
        // this expression in reasonable time" error.
        var stepSize = learningRate * sqrt(1 - pow(beta2, Float(step)))
        stepSize = stepSize / (1 - pow(beta1, Float(step)))
        // Update `Tensor<Float>` & `Tensor<Double>` variables.
        for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            firstMoments[keyPath: kp] =
                (beta1 * firstMoments[keyPath: kp]) + (1 - beta1) * direction[keyPath: kp]
            infinityNorm[keyPath: kp] =
                max(beta2 * infinityNorm[keyPath: kp], abs(direction[keyPath: kp]))
            let biasCorrection = stepSize / (1 - pow(beta1, Float(step)))
            model[keyPath: kp] -=
                biasCorrection * firstMoments[keyPath: kp]
                / (infinityNorm[keyPath: kp] + Float(self.epsilon))
        }
        for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
            firstMoments[keyPath: kp] =
                Double(beta1) * firstMoments[keyPath: kp]
                + Double(1 - beta2) * direction[keyPath: kp]
            infinityNorm[keyPath: kp] =
                max(Double(beta2) * infinityNorm[keyPath: kp], abs(direction[keyPath: kp]))
            let biasCorrection = Double(stepSize) / Double(1 - pow(beta1, Float(step)))
            model[keyPath: kp] -=
                biasCorrection * firstMoments[keyPath: kp]
                / (infinityNorm[keyPath: kp] + Double(self.epsilon))
        }
    }

    public func update(_ model: inout Model,
                       along direction: Model.TangentVector) {
        update(&model.allDifferentiableVariables, along: direction)
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
public class RMSProp<Model: Layer>: Optimizer
    where Model.AllDifferentiableVariables == Model.TangentVector {
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
    public var alpha: Model.AllDifferentiableVariables

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
        alpha = model.allDifferentiableVariables
        for kp in alpha.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            alpha[keyPath: kp].resetToZero()
        }
        for kp in alpha.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
            alpha[keyPath: kp].resetToZero()
        }
    }

    // TODO: Deprecate this when `Differentiable.AllDifferentiableVariables` is removed.
    public func update(_ model: inout Model.AllDifferentiableVariables,
                       along direction: Model.TangentVector) {
        step += 1
        let learningRate = self.learningRate * 1 / (1 + decay * Float(step))
        for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            alpha[keyPath: kp] =
                rho * alpha[keyPath: kp] + (1 - rho) * pow(direction[keyPath: kp], 2)
            model[keyPath: kp] -=
                learningRate * direction[keyPath: kp] / (sqrt(alpha[keyPath: kp]) + epsilon)
        }
        for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
            alpha[keyPath: kp] =
                Double(rho) * alpha[keyPath: kp] + Double(1 - rho) * pow(direction[keyPath: kp], 2)
            model[keyPath: kp] -=
                Double(learningRate) * direction[keyPath: kp] /
                (sqrt(alpha[keyPath: kp]) + Double(epsilon))
        }
    }

    public func update(_ model: inout Model,
                       along direction: Model.TangentVector) {
        update(&model.allDifferentiableVariables, along: direction)
    }
}

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

    // TODO: Deprecate this when `Differentiable.AllDifferentiableVariables` is removed.
    public func update(_ model: inout Model.AllDifferentiableVariables,
                       along direction: Model.TangentVector) {
        step += 1
        let learningRate = self.learningRate * 1 / (1 + decay * Float(step))
        velocity = momentum * velocity - direction * learningRate
        if nesterov {
            model.move(along: momentum * velocity - direction * learningRate)
        } else {
            model.move(along: velocity)
        }
    }

    public func update(_ model: inout Model,
                       along direction: Model.TangentVector) {
        update(&model.allDifferentiableVariables, along: direction)
    }
}


/// AdaGrad optimizer.
///
/// Individually adapts the learning rates of all model parameters by scaling them inversely proportional to
/// the square root of the sum of all the historical squared values of the gradient.
///
/// Reference: ["Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"](
/// http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
public class AdaGrad<Model: Layer>: Optimizer
    where Model.AllDifferentiableVariables == Model.TangentVector {
    public typealias Model = Model
    /// The learning rate.
    public var learningRate: Float
    /// The smoothing factor (ρ). Typical values are `0.5`, `0.9`, and `0.99`, for smoothing over 2,
    /// 10, and 100 examples, respectively.
    public var rho: Float
    /// A small scalar added to the denominator to improve numerical stability.
    public var epsilon: Float
    /// The alpha values for all model differentiable variables.
    public var alpha: Model.AllDifferentiableVariables

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

        alpha = model.allDifferentiableVariables
        for kp in alpha.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            alpha[keyPath: kp].resetToZero()
        }
        for kp in alpha.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
            alpha[keyPath: kp].resetToZero()
        }
    }

    // TODO: Deprecate this when `Differentiable.AllDifferentiableVariables` is removed.
    public func update(_ model: inout Model.AllDifferentiableVariables,
                       along direction: Model.TangentVector) {
        for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            alpha[keyPath: kp] = rho + direction[keyPath: kp].squared()
            model[keyPath: kp] -=
                learningRate * direction[keyPath: kp] / (sqrt(alpha[keyPath: kp] + epsilon))
        }
        for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
            alpha[keyPath: kp] = Double(rho) + direction[keyPath: kp].squared()
            model[keyPath: kp] -=
                Double(learningRate) * direction[keyPath: kp] /
                (sqrt(alpha[keyPath: kp] + Double(epsilon)))
        }
    }

    public func update(_ model: inout Model,
                       along direction: Model.TangentVector) {
        update(&model.allDifferentiableVariables, along: direction)
    }
}

/// ADADELTA optimizer.
///
/// ADADELTA is a more robust extension of AdaGrad. ADADELTA adapts learning rates based on a moving
/// window of gradient updates rather accumulating all past gradients. ADADELTA can continue to
/// learn even after many update steps.
/// 
/// Reference: ["ADADELTA: An Adaptive Learning Rate Method"](https://arxiv.org/abs/1212.5701)
public class AdaDelta<Model: Layer>: Optimizer
    where Model.AllDifferentiableVariables == Model.TangentVector {
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
    public var averageSquared: Model.TangentVector
    /// The accumulated parameter updates.
    public var accumulatedDelta: Model.TangentVector

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

        averageSquared = model.allDifferentiableVariables
        accumulatedDelta = model.allDifferentiableVariables
        
        for kp in averageSquared.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            averageSquared[keyPath: kp].resetToZero()
            accumulatedDelta[keyPath: kp].resetToZero()
        }
        for kp in averageSquared.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
            averageSquared[keyPath: kp].resetToZero()
            accumulatedDelta[keyPath: kp].resetToZero()
        }
    }

    // TODO: Deprecate this when `Differentiable.AllDifferentiableVariables` is removed.
    public func update(_ model: inout Model.AllDifferentiableVariables,
                       along direction: Model.AllDifferentiableVariables) {
        step += 1
        let learningRate = self.learningRate / (1 + decay * Float(step))
        
        // Update `Tensor<Float>` and `Tensor<Double>` variables.
        for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            averageSquared[keyPath: kp] *= rho
            averageSquared[keyPath: kp] +=
                (1 - rho) * (direction[keyPath: kp] * direction[keyPath: kp])
            var stepSize = direction[keyPath: kp] *
                sqrt(accumulatedDelta[keyPath: kp] + epsilon)
            stepSize /= sqrt(averageSquared[keyPath: kp] + epsilon)
            model[keyPath: kp] -= learningRate * stepSize
            accumulatedDelta[keyPath: kp] *= rho
            accumulatedDelta[keyPath: kp] += (1 - rho) * stepSize.squared()
        }
        for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
            averageSquared[keyPath: kp] *= Double(rho)
            averageSquared[keyPath: kp] +=
                (1 - Double(rho)) * (direction[keyPath: kp] * direction[keyPath: kp])
            var stepSize = direction[keyPath: kp] *
                sqrt(accumulatedDelta[keyPath: kp] + Double(epsilon))
            stepSize /= sqrt(averageSquared[keyPath: kp] + Double(epsilon))
            model[keyPath: kp] -= Double(learningRate) * stepSize
            accumulatedDelta[keyPath: kp] *= Double(rho)
            accumulatedDelta[keyPath: kp] += (1 - Double(rho)) * stepSize.squared()
        }
    }

    public func update(_ model: inout Model,
                       along direction: Model.TangentVector) {
        update(&model.allDifferentiableVariables, along: direction)
    }
}

// MARK: - Manifold optimizers

/// A Riemann manifold stochastic gradient descent (SGD) optimizer.
public class RiemannSGD<Model: Differentiable>: Optimizer
    where Model.TangentVector: VectorProtocol,
          Model.TangentVector.VectorSpaceScalar: FloatingPoint {
    public typealias Scalar = Model.TangentVector.VectorSpaceScalar
    /// The learning rate.
    public var learningRate: Model.TangentVector.VectorSpaceScalar

    public init(learningRate: Model.TangentVector.VectorSpaceScalar) {
        self.learningRate = learningRate
    }

    public convenience init(
        for _: __shared Model,
        learningRate: Scalar
    ) {
        self.init(learningRate: learningRate)
    }

    public func update(_ model: inout Model.AllDifferentiableVariables,
                       along direction: Model.TangentVector) {
        model.move(along: (.zero - direction).scaled(by: learningRate))
    }
}
