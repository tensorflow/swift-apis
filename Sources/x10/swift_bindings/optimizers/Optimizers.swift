// Copyright 2020 TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TensorFlow

fileprivate func l2Norm(_ x: Tensor<Float>) -> Tensor<Float> {
  return sqrt(x.squared().sum())
}

extension ParameterGroupOptimizerBuilder {
  /// Applies a sgdStep with momentum to the current parameter group optimization.
  public mutating func sgdStep(
    nesterov: Bool, mom: GlobalAccessor, lr: GlobalAccessor, velocity: StateAccessor
  ) {
    if nesterov {
      appendCallback { (state: inout OptimizerWeightStepState, optState: inout OptimizerState) in
        state.step = state[mom] * optState[state, velocity] - state.grad * state[lr]
      }
    } else {
      appendCallback { (state: inout OptimizerWeightStepState, optState: inout OptimizerState) in
        state.step = optState[state, velocity]
      }
    }
  }

  /// Computes the clippedTrustRatio (used in LARS).
  public mutating func clippedTrustRatio(
    trustCoefficient: GlobalAccessor,
    epsilon: GlobalAccessor, weightDecay: GlobalAccessor
  ) -> LocalAccessor {
    let trustRatio = self[local: "trustRatio"]
    let one = self.makeParameter("one", 1.0)
    appendCallback { (state: inout OptimizerWeightStepState, optState: inout OptimizerState) in
      let paramNorm = l2Norm(state.weight)
      let gradNorm = l2Norm(state.grad)
      let denom = gradNorm + state[weightDecay] * paramNorm
      let trustRatioTensor = state[trustCoefficient] * paramNorm / (denom + state[epsilon])
      state[trustRatio] = _Raw.select(
        condition: (gradNorm + paramNorm) .> 0, t: trustRatioTensor, e: state[one])
    }
    return trustRatio
  }

  /// Scales the gradient by the trustRatio (used in LARS).
  public mutating func scaleGradByTrustRatio(trustRatio: LocalAccessor) {
    appendCallback { (state: inout OptimizerWeightStepState, optState: inout OptimizerState) in
      state.grad = state.grad * state[trustRatio]
    }
  }

  /// Applies weight decay scaling to the gradient.
  public mutating func scaleGradient(byWeightDecay weightDecay: GlobalAccessor) {
    appendCallback { (state: inout OptimizerWeightStepState, optState: inout OptimizerState) in
      state.grad = state.grad + state.weight * state[weightDecay]
    }
  }

  /// Recomputes the velocity parameter based on the new gradient (scaled by the learning rate).
  public mutating func updateVelocity(
    mom: GlobalAccessor, lr: GlobalAccessor, velocity: StateAccessor
  ) {
    appendCallback { (state: inout OptimizerWeightStepState, optState: inout OptimizerState) in
      optState[state, velocity] =
        state[mom] * optState[state, velocity] - state.grad * state[lr]
    }
  }
}

/// Builds a per-weight optimizer for LARS (https://arxiv.org/pdf/1708.03888.pdf).
public func makeLARS(
  learningRate: Float = 0.01,
  momentum: Float = 0.9,
  trustCoefficient: Float = 0.001,
  nesterov: Bool = false,
  epsilon: Float = 0.0,
  weightDecay: Float = 0.0
) -> ParameterGroupOptimizer {
  var b = ParameterGroupOptimizerBuilder()
  let trustCoefficient = b.makeParameter("trustCoefficient", trustCoefficient)
  let lr = b.makeParameter("learningRate", learningRate)
  let mom = b.makeParameter("mom", momentum)
  let epsilon = b.makeParameter("epsilon", epsilon)
  let wd = b.makeParameter("weightDecay", weightDecay)
  let trustRatio = b.clippedTrustRatio(
    trustCoefficient: trustCoefficient, epsilon: epsilon, weightDecay: wd)
  if weightDecay != 0 { b.scaleGradient(byWeightDecay: wd) }
  b.scaleGradByTrustRatio(trustRatio: trustRatio)
  let velocity = b[state: "velocity"]
  b.updateVelocity(mom: mom, lr: lr, velocity: velocity)
  b.sgdStep(nesterov: nesterov, mom: mom, lr: lr, velocity: velocity)
  return b.makeOptimizer()
}

/// Builds a SGD based per-weight optimizer.
public func makeSGD(
  learningRate: Float = 0.01,
  momentum: Float = 0,
  weightDecay: Float = 0,
  nesterov: Bool = false
) -> ParameterGroupOptimizer {
  precondition(learningRate >= 0, "Learning rate must be non-negative")
  precondition(momentum >= 0, "Momentum must be non-negative")
  // TODO(parkers): Shorthand syntax: (["lr": learningRate, "mom": momentum, "weightDecay": weightDecay]) ??
  var b = ParameterGroupOptimizerBuilder()
  let lr = b.makeParameter("learningRate", learningRate)
  let mom = b.makeParameter("mom", momentum)
  let wd = b.makeParameter("weightDecay", weightDecay)
  if weightDecay != 0 { b.scaleGradient(byWeightDecay: wd) }
  let velocity = b[state: "velocity"]
  b.updateVelocity(mom: mom, lr: lr, velocity: velocity)
  b.sgdStep(nesterov: nesterov, mom: mom, lr: lr, velocity: velocity)
  return b.makeOptimizer()
}
