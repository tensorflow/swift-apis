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

/// Adam optimizer.
///
/// Reference: ["Adam - A Method for Stochastic Optimization"](
/// https://arxiv.org/abs/1412.6980v8)
public class Adam<Model: Differentiable>: Optimizer
    where Model.TangentVector: VectorProtocol & PointwiseMultiplicative & ElementaryFunctions,
          Model.TangentVector.VectorSpaceScalar == Float {
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
        update(&model.allDifferentiableVariables, along: direction)
    }

    // TODO: Deprecate this when `Differentiable.AllDifferentiableVariables` is removed.
    public func update(
        _ model: inout Model.AllDifferentiableVariables,
        along direction: Model.TangentVector
    ) {
        self.step += 1
        let step = Float(self.step)
        let learningRate = self.learningRate * 1 / (1 + decay * step)
        // Note: `stepSize` and `secondMoments` are split into two lines to avoid the "compiler is 
        // unable to type-check this expression in reasonable time" error.
        var stepSize = learningRate * sqrt(1 - pow(beta2, step))
        stepSize = stepSize / (1 - pow(beta1, step))
        firstMoments = firstMoments * beta1 + direction * (1 - beta1)
        secondMoments = secondMoments * beta2
        secondMoments += direction .* direction * (1 - beta2)
        let denominator = Model.TangentVector.sqrt(secondMoments) + epsilon
        model.move(along: -stepSize * firstMoments ./ denominator)
    }
}

/// AdaMax optimizer.
///
/// A variant of Adam based on the infinity-norm.
///
/// Reference: Section 7 of ["Adam - A Method for Stochastic Optimization"](
/// https://arxiv.org/abs/1412.6980v8)
public class AdaMax<Model: Differentiable & KeyPathIterable>: Optimizer
    where Model.TangentVector: VectorProtocol & PointwiseMultiplicative & 
                               ElementaryFunctions & KeyPathIterable,
          Model.TangentVector.VectorSpaceScalar == Float,
          Model.AllDifferentiableVariables == Model.TangentVector {
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
    public var firstMoments: Model.TangentVector = .zero
    /// The exponentially weighted infinity norm of the weights.
    public var infinityNorm: Model.TangentVector = .zero

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
    }

    public func update(_ model: inout Model, along direction: Model.TangentVector) {
        update(&model.allDifferentiableVariables, along: direction)
    }

    // TODO: Deprecate this when `Differentiable.AllDifferentiableVariables` is removed.
    public func update(
        _ model: inout Model.AllDifferentiableVariables,
        along direction: Model.TangentVector
    ) {
        self.step += 1
        let step = Float(self.step)
        let learningRate = self.learningRate * 1 / (1 + decay * step)
        // Note: `stepSize` is split into two lines to avoid the "compiler is unable to type-check
        // this expression in reasonable time" error.
        var stepSize = learningRate * sqrt(1 - pow(beta2, step))
        stepSize = stepSize / (1 - pow(beta1, step))
        firstMoments = firstMoments * beta1 + direction * (1 - beta1)

        // Update `infinityNorm` using a key path approach because `max(_:_:)` cannot be 
        // currently applied in a simpler manner.
        for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            infinityNorm[keyPath: kp] = max(
                beta2 * infinityNorm[keyPath: kp], abs(direction[keyPath: kp]))
        }
        for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
            infinityNorm[keyPath: kp] = max(
                Double(beta2) * infinityNorm[keyPath: kp], abs(direction[keyPath: kp]))
        }

        let denominator = infinityNorm + epsilon
        model.move(along: -stepSize * firstMoments ./ denominator)
    }
}

/// AMSGrad optimizer.
///
/// This algorithm is a modification of Adam with better convergence properties when close to local
/// optima.
///
/// Reference: ["On the Convergence of Adam and Beyond"](
/// https://openreview.net/pdf?id=ryQu7f-RZ)
public class AMSGrad<Model: Differentiable & KeyPathIterable>: Optimizer
    where Model.TangentVector: VectorProtocol & PointwiseMultiplicative & 
                               ElementaryFunctions & KeyPathIterable,
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
    /// The maximum of the second moments of the weights.
    public var secondMomentsMax: Model.TangentVector = .zero

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
        update(&model.allDifferentiableVariables, along: direction)
    }

    // TODO: Deprecate this when `Differentiable.AllDifferentiableVariables` is removed.
    public func update(
        _ model: inout Model.AllDifferentiableVariables,
        along direction: Model.TangentVector
    ) {
        self.step += 1
        let step = Float(self.step)
        let beta1Power = pow(beta1, step)
        let beta2Power = pow(beta2, step)
        let learningRate = self.learningRate * 1 / (1 + decay * step)
        // Note: `stepSize` and `secondMoments` are split into two lines to avoid the "compiler is 
        // unable to type-check this expression in reasonable time" error.
        var stepSize = learningRate * sqrt(1 - pow(beta2Power, step))
        stepSize = stepSize / (1 - pow(beta1Power, step))
        firstMoments = firstMoments * beta1 + direction * (1 - beta1)
        secondMoments = secondMoments * beta2
        secondMoments += direction .* direction * (1 - beta2)

        // Update `secondMomentsMax` using a key path approach because `max(_:_:)` cannot be 
        // currently applied in a simpler manner.
        for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            secondMomentsMax[keyPath: kp] = max(
                secondMomentsMax[keyPath: kp], secondMoments[keyPath: kp])
        }
        for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
            secondMomentsMax[keyPath: kp] = max(
                secondMomentsMax[keyPath: kp], secondMoments[keyPath: kp])
        }

        let denominator = Model.TangentVector.sqrt(secondMomentsMax) + epsilon
        model.move(along: -stepSize * firstMoments ./ denominator)
    }
}
