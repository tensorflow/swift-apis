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

/// A stochastic gradient descent (SGD) optimizer.
///
/// Implements the stochastic gradient descent algorithm with support for momentum, learning rate
/// decay, and Nesterov momentum. Momentum and Nesterov momentum (a.k.a. the Nesterov accelerated
/// gradient method) are first-order optimization methods that can improve the training speed and
/// convergence rate of gradient descent.
///
/// References:
/// - ["A Stochastic Approximation Method"](
/// https://projecteuclid.org/euclid.aoms/1177729586) (Robbins and Monro, 1951)
/// - ["On the Stochastic Approximation Method of Robbins and Monro"](
/// https://projecteuclid.org/euclid.aoms/1177729391) (Wolfowitz, 1952)
/// - ["Stochastic Estimation of the Maximum of a Regression Function"](
/// https://projecteuclid.org/euclid.aoms/1177729392) (Kiefer and Wolfowitz, 1952)
/// - ["Some methods of speeding up the convergence of iteration method"](
/// https://vsokolov.org/courses/750/2018/files/polyak64.pdf) (Polyak, 1964)
/// - ["A method for unconstrained convex minimization problem with the rate of
/// convergence"](http://mpawankumar.info/teaching/cdt-big-data/nesterov83.pdf)
/// (Nesterov, 1983)
public class SGD<Model: Differentiable>: Optimizer
where
  Model.TangentVector: VectorProtocol & ElementaryFunctions & KeyPathIterable,
  Model.TangentVector.VectorSpaceScalar == Float
{
  public typealias Model = Model
  /// The learning rate.
  public var learningRate: Float
  /// The momentum factor. It accelerates stochastic gradient descent in the relevant direction and
  /// dampens oscillations.
  public var momentum: Float
  /// The learning rate decay.
  public var decay: Float
  /// Use Nesterov momentum if true.
  public var nesterov: Bool
  /// The velocity state of the model.
  public var velocity: Model.TangentVector = .zero
  /// The set of steps taken.
  public var step: Int = 0

  /// Creates an instance for `model`.
  ///
  /// - Parameters:
  ///   - learningRate: The learning rate. The default value is `0.01`.
  ///   - momentum: The momentum factor that accelerates stochastic gradient descent in the relevant
  ///     direction and dampens oscillations. The default value is `0`.
  ///   - decay: The learning rate decay. The default value is `0`.
  ///   - nesterov: Use Nesterov momentum iff `true`. The default value is `true`.
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
      model.move(by: velocity.scaled(by: momentum) - direction.scaled(by: learningRate))
    } else {
      model.move(by: velocity)
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
