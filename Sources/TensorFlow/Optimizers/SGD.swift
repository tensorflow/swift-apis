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

/// Stochastic gradient descent (SGD) optimizer.
///
/// Implements the stochastic gradient descent optimizer algorithm with support 
/// for momentum, learning rate decay, and Nesterov momentum. Momentum and 
/// Nesterov momentum (a.k.a. the Nesterov accelerated gradient method) are 
/// first-order optimization methods that can improve the training speed and the
/// convergence rate of gradient descent.
///
/// Reference: ["A Stochastic Approximation Method"](
/// https://projecteuclid.org/euclid.aoms/1177729586) (Robbins and Monro, 1951), 
/// ["On the Stochastic Approximation Method of Robbins and Monro"](
/// https://projecteuclid.org/euclid.aoms/1177729391) (Wolfowitz, 1952) and
/// ["Stochastic Estimation of the Maximum of a Regression Function"](
/// https://projecteuclid.org/euclid.aoms/1177729392) (Kiefer and Wolfowitz, 1952), 
/// ["Some methods of speeding up the convergence of iteration method"](
/// https://vsokolov.org/courses/750/2018/files/polyak64.pdf) (Polyak, 1964),
/// ["A method for unconstrained convex minimization problem with the rate of 
/// convergence"](http://mpawankumar.info/teaching/cdt-big-data/nesterov83.pdf) 
/// (Nesterov, 1983).
/// 
/// - Parameters:
///     - learningRate: A float. The learning rate (default value: 0.01).
///     - momentum: The momentum factor. It accelerates stochastic gradient 
///     descent in the relevant direction and dampens oscillations (default 
///     value: 0).
///     - decay: A float. The learning rate decay (default value: 0).
///     - nesterov: A boolean. The Nesterov’s accelerated gradient method. 
///     (default value: true).
///
/// ### Examples: ###
///  
/// // - Transfer learning from pre-trained VGG19 weights:
/// 
/// ````
/// ...
/// // Define the base model architecture of VGG19.
/// var vgg19 = VGG19 { ... }
/// // Define new classifier layers.
/// struct Classifier: Layer { ... }
/// ...
/// // Initialize the base model.
/// let vgg = VGG19()
/// // Instantiate the added classifier layers that you're adding to the base model.
/// var transferLearningModel = Classifier()
/// // Define the SGD optimizer for the network with a learning rate set to  1e-3 
/// // and the Nesterov momentum enabled.
/// let sgdOptimizer = SGD(for: transferLearningModel, learningRate: 1e-3, nesterov: true)
/// ...
/// // Start the training loop over a certain number of epochs.
/// for epoch in 1...epochCount {
///     ...
///     for i in batchCount {
///         ...
///         // Define a feature extractor. 
///         let features = vgg(someBatchSize)
///         // Implementing the gradient descent with the AdaDelta optimizer:
///         // Compute the loss and gradient.
///         let (loss, 𝛁transferLearningModel = valueWithgradient(at: transferLearningModel) {
///             transferLearningModel -> Tensor<Float> in
///             ...
///             return loss
///         }
///         // Update the differentiable variables of the model along the gradients
///         // (`𝛁transferLearningModel`) with the SGD optimizer.
///         sgdOptimizer.update(&transferLearningModel, along: 𝛁transferLearningModel)
///     }
///     ...
/// }
/// ````
public class SGD<Model: Differentiable>: Optimizer
where
  Model.TangentVector: VectorProtocol & ElementaryFunctions & KeyPathIterable,
  Model.TangentVector.VectorSpaceScalar == Float
{
  public typealias Model = Model
  /// The learning rate.
  public var learningRate: Float
  /// The momentum factor. It accelerates stochastic gradient descent in the relevant direction
  /// and dampens oscillations.
  public var momentum: Float
  /// The learning rate decay.
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
    precondition(decay >= 0, "Learning rate decay must be non-negative")

    self.learningRate = learningRate
    self.momentum = momentum
    self.decay = decay
    self.nesterov = nesterov
  }

  public func update(_ model: inout Model, along direction: Model.TangentVector) {
    step += 1
    let learningRate = self.learningRate * 1 / (1 + decay * Float(step))
    velocity = velocity.scaled(by: momentum) - direction.scaled(by: learningRate)
    if nesterov {
      model.move(along: velocity.scaled(by: momentum) - direction.scaled(by: learningRate))
    } else {
      model.move(along: velocity)
    }
  }

  public required init(copying other: SGD, to device: Device) {
    learningRate = other.learningRate
    momentum = other.momentum
    decay = other.decay
    nesterov = other.nesterov
    velocity = .init(copying: other.velocity, to: device)
    step = other.step
  }
}
