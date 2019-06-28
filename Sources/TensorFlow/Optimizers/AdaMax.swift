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

fileprivate extension Tensor where Scalar: Numeric {
    mutating func resetToZero() {
        self = Tensor(zeros: shape)
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
