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

/// Reciprocal square root learning rate decay schedule.
///
/// The decayed learning rate is computed as follows:
/// ```
/// decay = decayFactor / sqrt(max(step, decayThreshold))
/// decayedLearningRate = learningRate * ((1 - lowerBound) * decay + lowerBound)
/// ```
public struct RSqrtLearningRateDecay<
    Scalar: FloatingPoint & ElementaryFunctions
>: LearningRateSchedule {
    public let decayFactor: Scalar
    public let decayThreshold: Scalar
    public let lowerBound: Scalar
    public let startStep: UInt64

    /// Creates a new reciprocal square root learning rate decay schedule.
    ///
    /// - Parameters:
    ///   - decayFactor: Decay factor.
    ///   - decayThreshold: Decay threshold.
    ///   - lowerBound: Minimum decayed learning rate value as a fraction of the original learning
    ///     rate value.
    ///   - startStep: Step after which to start decaying the learning rate.
    @inlinable
    public init(
        decayFactor: Scalar,
        decayThreshold: Scalar,
        lowerBound: Scalar = Scalar(0),
        startStep: UInt64 = 0
    ) {
        self.decayFactor = decayFactor
        self.decayThreshold = decayThreshold
        self.lowerBound = lowerBound
        self.startStep = startStep
    }

    @inlinable
    public func callAsFunction(step: UInt64, learningRate: Scalar) -> Scalar {
        if step < startStep { return learningRate }
        let step = step - startStep
        let decay = decayFactor / Scalar.sqrt(max(Scalar(step), decayThreshold))
        return learningRate * ((1 - lowerBound) * decay + lowerBound)
    }
}

/// Cosine learning rate decay schedule.
///
/// The decayed learning rate is computed as follows:
/// ```
/// decay = 0.5 * (1 + cos(pi * min(step, cycleStepCount) / cycleStepCount))
/// decayedLearningRate = learningRate * ((1 - lowerBound) * decay + lowerBound)
/// ```
public struct CosineLearningRateDecay<
    Scalar: FloatingPoint & ElementaryFunctions
>: LearningRateSchedule {
    public let cycleStepCount: UInt64
    public let lowerBound: Scalar
    public let startStep: UInt64

    /// Creates a new cosine learning rate decay schedule.
    ///
    /// - Parameters:
    ///   - cycleStepCount: Cosine decay cycle in terms of number of steps.
    ///   - lowerBound: Minimum decayed learning rate value as a fraction of the original learning
    ///     rate value.
    ///   - startStep: Step after which to start decaying the learning rate.
    @inlinable
    public init(
        cycleStepCount: UInt64,
        lowerBound: Scalar = Scalar(0),
        startStep: UInt64 = 0
    ) {
        self.cycleStepCount = cycleStepCount
        self.lowerBound = lowerBound
        self.startStep = startStep
    }

    @inlinable
    public func callAsFunction(step: UInt64, learningRate: Scalar) -> Scalar {
        if step < startStep { return learningRate }
        let step = step - startStep
        let cosine = Scalar.cos(Scalar(min(step, cycleStepCount)))
        let decay = (1 + cosine) * Scalar.pi / Scalar(2 * cycleStepCount)
        return learningRate * ((1 - lowerBound) * decay + lowerBound)
    }
}

/// Cycle-linear 10x learning rate decay schedule.
///
/// The decayed learning rate is computed as follows:
/// ```
/// cyclePosition = 1 - abs((step % (2 * cycleStepCount) - cycleStepCount) / cycleStepCount)
/// decay = (0.1 + cyclePosition) * 3
/// decayedLearningRate = learningRate * ((1 - lowerBound) * decay + lowerBound)
/// ```
public struct CycleLinear10xLearningRateDecay<Scalar: FloatingPoint>: LearningRateSchedule {
    public let cycleStepCount: UInt64
    public let lowerBound: Scalar
    public let startStep: UInt64

    /// Creates a new cycle-linear 10x learning rate decay schedule.
    ///
    /// - Parameters:
    ///   - cycleStepCount: Cycle-linear 10x decay cycle in terms of number of steps.
    ///   - lowerBound: Minimum decayed learning rate value as a fraction of the original learning
    ///     rate value.
    ///   - startStep: Step after which to start decaying the learning rate.
    @inlinable
    public init(
        cycleStepCount: UInt64,
        lowerBound: Scalar = Scalar(0),
        startStep: UInt64 = 0
    ) {
        self.cycleStepCount = cycleStepCount
        self.lowerBound = lowerBound
        self.startStep = startStep
    }

    @inlinable
    public func callAsFunction(step: UInt64, learningRate: Scalar) -> Scalar {
        if step < startStep { return learningRate }
        let step = step - startStep
        let ratio = Scalar((step % (2 * cycleStepCount) - cycleStepCount)) / Scalar(cycleStepCount)
        let cyclePosition = 1 - abs(ratio)
        let decay = (1 / Scalar(10) + cyclePosition) * 3 // 10x difference in each cycle (0.3 - 3).
        return learningRate * ((1 - lowerBound) * decay + lowerBound)
    }
}

/// Linear learning rate warm-up schedule.
///
/// For the first `warmUpStepCount` steps the learning rate is multiplied with:
/// ```
/// warmUpOffset + ((1 - warmUpOffset) / warmUpStepCount) * step
/// ```
///
/// - Source: [Attention is All You Need (Section 5.3)](https://arxiv.org/pdf/1706.03762.pdf).
public struct LinearLearningRateWarmUp<Scalar: FloatingPoint>: LearningRateSchedule {
    public let warmUpStepCount: UInt64
    public let warmUpOffset: Scalar

    /// Creates a new linear learning rate warm-up schedule.
    ///
    /// - Parameters:
    ///   - warmUpStepCount: Number of warm-up steps.
    ///   - warmUpOffset: Linear schedule offset.
    @inlinable
    public init(warmUpStepCount: UInt64, warmUpOffset: Scalar) {
        self.warmUpStepCount = warmUpStepCount
        self.warmUpOffset = warmUpOffset
    }

    @inlinable
    public func callAsFunction(step: UInt64, learningRate: Scalar) -> Scalar {
        if step >= warmUpStepCount { return learningRate }
        let factor = warmUpOffset + ((1 - warmUpOffset) / Scalar(warmUpStepCount)) * Scalar(step)
        return learningRate * factor
    }
}

/// Exponential learning rate warm-up schedule.
///
/// For the first `warmUpStepCount` steps the learning rate is multiplied with:
/// ```
/// exp(log(warmUpFactor) / step) ^ (warmUpStepCount - step)
/// ```
///
/// - Source: [Attention is All You Need (Section 5.3)](https://arxiv.org/pdf/1706.03762.pdf).
public struct ExponentialLearningRateWarmUp<
    Scalar: FloatingPoint & ElementaryFunctions
>: LearningRateSchedule {
    public let warmUpStepCount: UInt64
    public let warmUpFactor: Scalar

    /// Creates a new linear learning rate warm-up schedule.
    ///
    /// - Parameters:
    ///   - warmUpStepCount: Number of warm-up steps.
    ///   - warmUpFactor: Warm-up learning rate scaling factor.
    @inlinable
    public init(warmUpStepCount: UInt64, warmUpFactor: Scalar) {
        self.warmUpStepCount = warmUpStepCount
        self.warmUpFactor = warmUpFactor
    }

    @inlinable
    public func callAsFunction(step: UInt64, learningRate: Scalar) -> Scalar {
        if step >= warmUpStepCount { return learningRate }
        let base = Scalar.exp(Scalar.log(warmUpFactor) / Scalar(warmUpStepCount))
        let factor = Scalar.pow(base, Scalar(warmUpStepCount - step))
        return learningRate * factor
    }
}
