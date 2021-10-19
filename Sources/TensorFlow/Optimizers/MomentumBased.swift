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

import _Differentiation
#if TENSORFLOW_USE_STANDARD_TOOLCHAIN
import Numerics
#endif

/// A RMSProp optimizer.
///
/// Implements the RMSProp optimization algorithm. RMSProp is a form of stochastic gradient descent
/// where the gradients are divided by a running average of their recent magnitude. RMSProp keeps a
/// moving average of the squared gradient for each weight.
///
/// References:
/// - ["Lecture 6.5 - rmsprop: Divide the gradient by a running average
/// of its recent magnitude"](
/// http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) 
/// (Tieleman and Hinton, 2012)
/// - ["Generating Sequences With Recurrent Neural Networks"](
/// https://arxiv.org/abs/1308.0850) (Graves, 2013)
public class RMSProp<Model: Differentiable>: Optimizer
where
  Model.TangentVector: VectorProtocol & PointwiseMultiplicative
    & ElementaryFunctions & KeyPathIterable,
  Model.TangentVector.VectorSpaceScalar == Float
{
  public typealias Model = Model
  /// The learning rate.
  public var learningRate: Float
  /// The gradient moving average decay factor.
  public var rho: Float
  /// A small scalar added to the denominator to improve numerical stability.
  public var epsilon: Float
  /// The learning rate decay.
  public var decay: Float
  /// The step count.
  public var step: Float = 0
  /// The alpha values for all model differentiable variables.
  public var alpha: Model.TangentVector = .zero

  /// Creates an instance for `model`.
  ///
  /// - Parameters:
  ///   - learningRate: The learning rate. The default value is `1e-3`.
  ///   - rho: The gradient moving average decay factor. The default value is `0.9`.
  ///   - epsilon: A small scalar added to the denominator to improve numerical stability. The
  ///     default value is `1e-8`.
  ///   - decay: The learning rate decay. The default value is `0`.
  public init(
    for model: __shared Model,
    learningRate: Float = 1e-3,
    rho: Float = 0.9,
    epsilon: Float = 1e-8,
    decay: Float = 0
  ) {
    precondition(learningRate >= 0, "Learning rate must be non-negative")
    precondition(rho >= 0, "Rho must be non-negative")
    precondition(decay >= 0, "Learning rate decay must be non-negative")

    self.learningRate = learningRate
    self.rho = rho
    self.epsilon = epsilon
    self.decay = decay
  }

  public func update(_ model: inout Model, along direction: Model.TangentVector) {
    step += 1
    let learningRate = self.learningRate * 1 / (1 + decay * Float(step))
    alpha = alpha.scaled(by: rho) + (direction .* direction).scaled(by: 1 - rho)
    let denominator = Model.TangentVector.sqrt(alpha).adding(epsilon)
    model.move(by: (direction ./ denominator).scaled(by: -learningRate))
  }

  public required init(copying other: RMSProp, to device: Device) {
    learningRate = other.learningRate
    rho = other.rho
    epsilon = other.epsilon
    decay = other.decay
    step = other.step
    alpha = .init(copying: other.alpha, to: device)
  }
}

/// An AdaGrad optimizer.
///
/// Implements the AdaGrad (adaptive gradient) optimization algorithm. AdaGrad has
/// parameter-specific learning rates, which are adapted relative to how frequently parameters
/// gets updated during training. Parameters that receive more updates have smaller learning rates.
///
/// AdaGrad individually adapts the learning rates of all model parameters by scaling them inversely
/// proportional to the square root of the running sum of squares of gradient norms.
///
/// Reference: ["Adaptive Subgradient Methods for Online Learning and Stochastic 
/// Optimization"](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) 
/// (Duchi et al, 2011)
public class AdaGrad<Model: Differentiable>: Optimizer
where
  Model.TangentVector: VectorProtocol & PointwiseMultiplicative
    & ElementaryFunctions & KeyPathIterable,
  Model.TangentVector.VectorSpaceScalar == Float
{
  public typealias Model = Model
  /// The learning rate.
  public var learningRate: Float
  /// A small scalar added to the denominator to improve numerical stability.
  public var epsilon: Float
  /// The running sum of squares of gradient norms.
  public var accumulator: Model.TangentVector

  /// Creates an instance for `model`.
  ///
  /// - Parameters:
  ///   - learningRate: The learning rate. The default value is `1e-3`.
  ///   - initialAccumulatorValue: The starting value for the running sum of squares of gradient
  ///     norms. The default value is `0.1`.
  ///   - epsilon: A small scalar added to the denominator to improve numerical stability. The
  ///     default value is `1e-8`.
  public init(
    for model: __shared Model,
    learningRate: Float = 1e-3,
    initialAccumulatorValue: Float = 0.1,
    epsilon: Float = 1e-8
  ) {
    precondition(learningRate >= 0, "Learning rate must be non-negative")
    precondition(
      initialAccumulatorValue >= 0, "The initial accumulator value must be non-negative.")

    self.learningRate = learningRate
    self.epsilon = epsilon
    self.accumulator = Model.TangentVector.one.scaled(by: initialAccumulatorValue)
  }

  public func update(_ model: inout Model, along direction: Model.TangentVector) {
    accumulator = accumulator + (direction .* direction)
    let denominator = Model.TangentVector.sqrt(accumulator).adding(epsilon)
    model.move(by: (direction ./ denominator).scaled(by: -learningRate))
  }

  public required init(copying other: AdaGrad, to device: Device) {
    learningRate = other.learningRate
    epsilon = other.epsilon
    accumulator = .init(copying: other.accumulator, to: device)
  }
}

/// An AdaDelta optimizer.
///
/// Implements the AdaDelta optimization algorithm. AdaDelta is a stochastic
/// gradient descent method based on the first order information. It adapts
/// learning rates based on a moving window of gradient updates, instead of
/// accumulating all past gradients. Thus, AdaDelta continues learning even
/// when many updates have been done. It adapts faster to changing dynamics of
/// the optimization problem space.
/// 
/// Reference: ["ADADELTA: An Adaptive Learning Rate Method"](
/// https://arxiv.org/abs/1212.5701) (Zeiler, 2012)
public class AdaDelta<Model: Differentiable>: Optimizer
where
  Model.TangentVector: VectorProtocol & PointwiseMultiplicative
    & ElementaryFunctions & KeyPathIterable,
  Model.TangentVector.VectorSpaceScalar == Float
{
  public typealias Model = Model
  /// The learning rate.
  public var learningRate: Float
  /// The decay factor, corresponding to the fraction of gradient to keep at each time step.
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

  /// Creates an instance for `model`.
  ///
  /// - Parameters:
  ///   - learningRate: The learning rate. The default value is `1`.
  ///   - rho: The decay factor. The default value is `0.95`.
  ///   - epsilon: A small scalar added to the denominator to improve numerical stability. The
  ///     default value is `1e-6`.
  ///   - decay: The learning rate decay. The defalut value is `0`.
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
    step += 1
    let learningRate = self.learningRate / (1 + decay * Float(step))
    averageSquared =
      averageSquared.scaled(by: rho) + (direction .* direction).scaled(by: 1 - rho)
    var stepSize = direction .* Model.TangentVector.sqrt(accumulatedDelta.adding(epsilon))
    stepSize ./= Model.TangentVector.sqrt(averageSquared.adding(epsilon))
    model.move(by: stepSize.scaled(by: -learningRate))
    accumulatedDelta =
      accumulatedDelta.scaled(by: rho) + (stepSize .* stepSize).scaled(by: 1 - rho)
  }

  public required init(copying other: AdaDelta, to device: Device) {
    learningRate = other.learningRate
    rho = other.rho
    epsilon = other.epsilon
    decay = other.decay
    step = other.step
    averageSquared = .init(copying: other.averageSquared, to: device)
    accumulatedDelta = .init(copying: other.accumulatedDelta, to: device)
  }
}

/// Adam optimizer.
///
/// Implements the Adam optimization algorithm. Adam is a stochastic gradient descent method that 
/// computes individual adaptive learning rates for different parameters from estimates of first- 
/// and second-order moments of the gradients.
///
/// Reference: ["Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980v8) 
/// (Kingma and Ba, 2014).
///
/// ### Examples: ###
///
/// - Train a simple reinforcement learning agent:
///
/// ````
/// ...
/// // Instantiate an agent's policy - approximated by the neural network (`net`) after defining it 
/// in advance.
/// var net = Net(observationSize: Int(observationSize), hiddenSize: hiddenSize, actionCount: actionCount)
/// // Define the Adam optimizer for the network with a learning rate set to 0.01.
/// let optimizer = Adam(for: net, learningRate: 0.01)
/// ...
/// // Begin training the agent (over a certain number of episodes).
/// while true {
/// ...
///     // Implementing the gradient descent with the Adam optimizer:
///     // Define the gradients (use withLearningPhase to call a closure under a learning phase).
///     let gradients = withLearningPhase(.training) {
///         TensorFlow.gradient(at: net) { net -> Tensor<Float> in
///             // Return a softmax (loss) function
///             return loss = softmaxCrossEntropy(logits: net(input), probabilities: target)
///         }
///     }
///     // Update the differentiable variables of the network (`net`) along the gradients with the Adam 
/// optimizer.
///     optimizer.update(&net, along: gradients)
///     ...
///     }
/// }
/// ````
///
/// - Train a generative adversarial network (GAN):
///
/// ````
/// ...
/// // Instantiate the generator and the discriminator networks after defining them.
/// var generator = Generator()
/// var discriminator = Discriminator()
/// // Define the Adam optimizers for each network with a learning rate set to 2e-4 and beta1 - to 0.5.
/// let adamOptimizerG = Adam(for: generator, learningRate: 2e-4, beta1: 0.5)
/// let adamOptimizerD = Adam(for: discriminator, learningRate: 2e-4, beta1: 0.5)
/// ...
/// Start the training loop over a certain number of epochs (`epochCount`).
/// for epoch in 1...epochCount {
///     // Start the training phase.
///     ...
///     for batch in trainingShuffled.batched(batchSize) {
///         // Implementing the gradient descent with the Adam optimizer:
///         // 1) Update the generator.
///         ...
///         let 𝛁generator = TensorFlow.gradient(at: generator) { generator -> Tensor<Float> in
///             ...
///             return loss
///             }
///         // Update the differentiable variables of the generator along the gradients (`𝛁generator`) 
///         // with the Adam optimizer.
///         adamOptimizerG.update(&generator, along: 𝛁generator)
///
///         // 2) Update the discriminator.
///         ...
///         let 𝛁discriminator = TensorFlow.gradient(at: discriminator) { discriminator -> Tensor<Float> in
///             ...
///             return loss
///         }
///         // Update the differentiable variables of the discriminator along the gradients (`𝛁discriminator`) 
///         // with the Adam optimizer.
///         adamOptimizerD.update(&discriminator, along: 𝛁discriminator)
///         }
/// }       
/// ````
public class Adam<Model: Differentiable>: Optimizer
where
  Model.TangentVector: VectorProtocol & PointwiseMultiplicative
    & ElementaryFunctions & KeyPathIterable,
  Model.TangentVector.VectorSpaceScalar == Float
{
  public typealias Model = Model
  /// The learning rate.
  public var learningRate: Float
  /// A coefficient used to calculate the first moments of the gradients.
  public var beta1: Float
  /// A coefficient used to calculate the second moments of the gradients.
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

  /// - Parameters:
  ///   - learningRate: The learning rate. The default value is `1e-3`.
  ///   - beta1: The exponential decay rate for the 1st moment estimates. The default value is `0.9`.
  ///   - beta2: The exponential decay rate for the 2nd moment estimates. The default value is `0.999`.
  ///   - epsilon: A small scalar added to the denominator to improve numerical stability.
  ///     The default value is `1e-8`.
  ///   - decay: The learning rate decay. The default value is `0`.
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
    let step = Float(self.step)
    let learningRate = self.learningRate * 1 / (1 + decay * step)
    // Note: `stepSize` is split into two lines to avoid the "compiler is unable to type-check
    // this expression in reasonable time" error.
    var stepSize = learningRate * sqrtf(1 - powf(beta2, step))
    stepSize = stepSize / (1 - powf(beta1, step))
    firstMoments = firstMoments.scaled(by: beta1) + direction.scaled(by: 1 - beta1)
    secondMoments =
      secondMoments.scaled(by: beta2) + (direction .* direction).scaled(by: 1 - beta2)
    let denominator = Model.TangentVector.sqrt(secondMoments).adding(epsilon)
    model.move(by: (firstMoments ./ denominator).scaled(by: -stepSize))
  }

  public required init(copying other: Adam, to device: Device) {
    learningRate = other.learningRate
    beta1 = other.beta1
    beta2 = other.beta2
    epsilon = other.epsilon
    decay = other.decay
    step = other.step
    firstMoments = .init(copying: other.firstMoments, to: device)
    secondMoments = .init(copying: other.secondMoments, to: device)
  }
}

/// AdaMax optimizer.
///
/// A variant of Adam based on the infinity-norm.
///
/// Reference: Section 7 of ["Adam - A Method for Stochastic Optimization"](
/// https://arxiv.org/abs/1412.6980v8)
public class AdaMax<Model: Differentiable & KeyPathIterable>: Optimizer
where
  Model.TangentVector: VectorProtocol & PointwiseMultiplicative & ElementaryFunctions
    & KeyPathIterable,
  Model.TangentVector.VectorSpaceScalar == Float
{
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
    step += 1
    let step = Float(self.step)
    let learningRate = self.learningRate * 1 / (1 + decay * step)
    let stepSize = learningRate / (1 - powf(beta1, step))
    firstMoments = firstMoments.scaled(by: beta1) + direction.scaled(by: 1 - beta1)

    // Update `infinityNorm` using a key path approach because `max(_:_:)` cannot be 
    // currently applied in a simpler manner.
    for kp in infinityNorm.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      infinityNorm[keyPath: kp] = max(
        beta2 * infinityNorm[keyPath: kp], abs(direction[keyPath: kp]))
    }
    for kp in infinityNorm.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
      infinityNorm[keyPath: kp] = max(
        Double(beta2) * infinityNorm[keyPath: kp], abs(direction[keyPath: kp]))
    }

    let denominator = infinityNorm.adding(epsilon)
    model.move(by: (firstMoments ./ denominator).scaled(by: -stepSize))
  }

  public required init(copying other: AdaMax, to device: Device) {
    learningRate = other.learningRate
    beta1 = other.beta1
    beta2 = other.beta2
    epsilon = other.epsilon
    decay = other.decay
    step = other.step
    firstMoments = .init(copying: other.firstMoments, to: device)
    infinityNorm = .init(copying: other.infinityNorm, to: device)
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
where
  Model.TangentVector: VectorProtocol & PointwiseMultiplicative & ElementaryFunctions
    & KeyPathIterable,
  Model.TangentVector.VectorSpaceScalar == Float
{
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
    step += 1
    let step = Float(self.step)
    let learningRate = self.learningRate * 1 / (1 + decay * step)
    // Note: `stepSize` is split into two lines to avoid the "compiler is unable to type-check
    // this expression in reasonable time" error.
    var stepSize = learningRate * sqrtf(1 - powf(beta2, step))
    stepSize = stepSize / (1 - powf(beta1, step))
    firstMoments = firstMoments.scaled(by: beta1) + direction.scaled(by: 1 - beta1)
    secondMoments =
      secondMoments.scaled(by: beta2) + (direction .* direction).scaled(by: 1 - beta2)

    // Update `secondMomentsMax` using a key path approach because `max(_:_:)` cannot be 
    // currently applied in a simpler manner.
    for kp in secondMomentsMax.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      secondMomentsMax[keyPath: kp] = max(
        secondMomentsMax[keyPath: kp], secondMoments[keyPath: kp])
    }
    for kp in secondMomentsMax.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
      secondMomentsMax[keyPath: kp] = max(
        secondMomentsMax[keyPath: kp], secondMoments[keyPath: kp])
    }

    let denominator = Model.TangentVector.sqrt(secondMomentsMax).adding(epsilon)
    model.move(by: (firstMoments ./ denominator).scaled(by: -stepSize))
  }

  public required init(copying other: AMSGrad, to device: Device) {
    learningRate = other.learningRate
    beta1 = other.beta1
    beta2 = other.beta2
    epsilon = other.epsilon
    decay = other.decay
    step = other.step
    firstMoments = .init(copying: other.firstMoments, to: device)
    secondMoments = .init(copying: other.secondMoments, to: device)
    secondMomentsMax = .init(copying: other.secondMomentsMax, to: device)
  }
}

/// RAdam optimizer.
/// 
/// Rectified Adam, a variant of Adam that introduces a term to rectify the adaptive learning rate
/// variance.
/// 
/// Reference: ["On the Variance of the Adaptive Learning Rate and Beyond"](
/// https://arxiv.org/pdf/1908.03265.pdf)
public class RAdam<Model: Differentiable>: Optimizer
where
  Model.TangentVector: VectorProtocol & PointwiseMultiplicative & ElementaryFunctions
    & KeyPathIterable,
  Model.TangentVector.VectorSpaceScalar == Float
{
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
    let step = Float(self.step)
    let beta1Power = powf(beta1, step)
    let beta2Power = powf(beta2, step)
    secondMoments =
      secondMoments.scaled(by: beta2) + (direction .* direction).scaled(by: 1 - beta2)
    firstMoments = firstMoments.scaled(by: beta1) + direction.scaled(by: 1 - beta1)
    // Compute maximum length SMA, bias-corrected moving average and approximate length.
    let N_sma_inf = 2 / (1 - beta2) - 1
    let N_sma_t = N_sma_inf - 2 * step * beta2Power / (1 - beta2Power)

    if N_sma_t >= 5 {
      // Compute bias-corrected second moments, rectification and adapted momentum.
      let secondMoments_h = Model.TangentVector.sqrt(secondMoments).adding(epsilon)
      let stepSize =
        sqrtf(
          (N_sma_t - 4) * (N_sma_t - 2) * N_sma_inf
            / ((N_sma_inf - 4) * (N_sma_inf - 2) * (N_sma_t))) * learningRate / (1 - beta1Power)
      model.move(
        by: (firstMoments ./ secondMoments_h).scaled(by: -stepSize * sqrtf(1 - beta2Power)))
    } else {
      // Update with un-adapted momentum.
      let stepSize = learningRate / (1 - beta1Power)
      model.move(by: firstMoments.scaled(by: -stepSize))
    }
  }

  public required init(copying other: RAdam, to device: Device) {
    learningRate = other.learningRate
    beta1 = other.beta1
    beta2 = other.beta2
    epsilon = other.epsilon
    decay = other.decay
    step = other.step
    firstMoments = .init(copying: other.firstMoments, to: device)
    secondMoments = .init(copying: other.secondMoments, to: device)
  }
}
