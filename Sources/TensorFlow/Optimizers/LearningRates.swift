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

/// Learning rate schedule that takes a learning rate as input, along with the current training
/// step and returns a modified learning rate (e.g., decayed).
public protocol LearningRateSchedule {
  associatedtype Scalar: FloatingPoint

  /// Returns the transformed value of `learningRate` for the specified training step.
  ///
  /// - Parameters:
  ///   - step: Training step.
  ///   - learningRate: Learning rate value to transform according to this schedule.
  func callAsFunction(step: UInt64, learningRate: Scalar) -> Scalar
}

/// Dummy learning rate schedule that represents no schedule being used. This is useful as a
/// default value whenever a learning rate schedule argument is used.
public struct FixedLearningRate<Scalar: FloatingPoint>: LearningRateSchedule {
  @inlinable
  public func callAsFunction(step: UInt64, learningRate: Scalar) -> Scalar {
    learningRate
  }
}

/// Linear learning rate decay schedule.
///
/// The decayed learning rate is computed as follows:
/// ```
/// decayed = learningRate + step * slope
/// decayedLearningRate = max(lowerBound * learningRate, decayed)
/// ```
public struct LinearLearningRateDecay<Scalar: FloatingPoint>: LearningRateSchedule {
  public let slope: Scalar
  public let lowerBound: Scalar
  public let startStep: UInt64

  /// Creates a new linear learning rate decay schedule.
  ///
  /// - Parameters:
  ///   - slope: Slope of the linear decay.
  ///   - lowerBound: Minimum decayed learning rate value as a fraction of the original learning
  ///     rate value.
  ///   - startStep: Step after which to start decaying the learning rate.
  @inlinable
  public init(slope: Scalar, lowerBound: Scalar = Scalar(0), startStep: UInt64 = 0) {
    self.slope = slope
    self.lowerBound = lowerBound
    self.startStep = startStep
  }

  @inlinable
  public func callAsFunction(step: UInt64, learningRate: Scalar) -> Scalar {
    if step < startStep { return learningRate }
    let step = step - startStep
    let decayed = learningRate + Scalar(step) * slope
    return max(lowerBound * learningRate, decayed)
  }
}

/// Exponential learning rate decay schedule.
///
/// The decayed learning rate is computed as follows:
/// ```
/// decay = decayRate ^ (step / decayStepCount)
/// decayedLearningRate = learningRate * ((1 - lowerBound) * decay + lowerBound)
/// ```
/// where if `staircase = true`, then `step / decayStepCount` uses integer division and the decayed
/// learning rate follows a staircase function.
public struct ExponentialLearningRateDecay<
  Scalar: FloatingPoint & ElementaryFunctions
>: LearningRateSchedule {
  public let decayRate: Scalar
  public let decayStepCount: UInt64
  public let staircase: Bool
  public let lowerBound: Scalar
  public let startStep: UInt64

  /// Creates a new exponential learning rate decay schedule.
  ///
  /// - Parameters:
  ///   - decayRate: Decay rate.
  ///   - decayStepCount: Decay step count.
  ///   - staircase: If `true`, the decay will occur at discrete intervals.
  ///   - lowerBound: Minimum decayed learning rate value as a fraction of the original learning
  ///     rate value.
  ///   - startStep: Step after which to start decaying the learning rate.
  @inlinable
  public init(
    decayRate: Scalar,
    decayStepCount: UInt64,
    staircase: Bool = false,
    lowerBound: Scalar = Scalar(0),
    startStep: UInt64 = 0
  ) {
    self.decayRate = decayRate
    self.decayStepCount = decayStepCount
    self.staircase = staircase
    self.lowerBound = lowerBound
    self.startStep = startStep
  }

  @inlinable
  public func callAsFunction(step: UInt64, learningRate: Scalar) -> Scalar {
    if step < startStep { return learningRate }
    let step = step - startStep
    let power = Scalar(step) / Scalar(decayStepCount)
    let decay = Scalar.pow(decayRate, staircase ? power.rounded(.down) : power)
    return learningRate * ((1 - lowerBound) * decay + lowerBound)
  }
}
