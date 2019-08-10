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

/// Learning rate schedule that takes the current training step as input and returns a learning
/// rate to be used for training.
public protocol LearningRate {
    associatedtype Scalar: FloatingPoint

    /// Returns the learning rate value for the specified training step.
    ///
    /// - Parameter step: Training step.
    func callAsFunction(forStep step: UInt64) -> Scalar
}

/// Dummy learning rate schedule that represents no schedule being used. This is useful as a
/// default value whenever a learning rate schedule argument is used.
public struct FixedLearningRate<Scalar: FloatingPoint>: LearningRate {
    public let value: Scalar

    @inlinable
    public init(_ value: Scalar) {
        self.value = value
    }

    @inlinable
    public func callAsFunction(forStep step: UInt64) -> Scalar {
        value
    }
}

extension FixedLearningRate: ExpressibleByFloatLiteral
where Scalar: _ExpressibleByBuiltinFloatLiteral {
    public typealias FloatLiteralType = Scalar

    public init(floatLiteral value: Scalar) {
        self.init(value)
    }
}

/// Linearly decayed learning rate.
///
/// The decayed learning rate is computed as follows:
/// ```swift
/// let initial = baseLearningRate(forStep: step)
/// let decayed = initial + step * slope
/// let decayedLearningRate = max(lowerBound * initial, decayed)
/// ```
public struct LinearlyDecayedLearningRate<BaseLearningRate: LearningRate>: LearningRate {
    public typealias Scalar = BaseLearningRate.Scalar

    public let baseLearningRate: BaseLearningRate
    public let slope: Scalar
    public let lowerBound: Scalar
    public let startStep: UInt64

    /// Creates a new linearly decayed learning rate.
    ///
    /// - Parameters:
    ///   - baseLearningRate: Learning rate to decay.
    ///   - slope: Slope of the linear decay.
    ///   - lowerBound: Minimum decayed learning rate value as a fraction of the original learning
    ///     rate value.
    ///   - startStep: Step after which to start decaying the learning rate.
    @inlinable
    public init(
        baseLearningRate: BaseLearningRate,
        slope: Scalar,
        lowerBound: Scalar = Scalar(0),
        startStep: UInt64 = 0
    ) {
        self.baseLearningRate = baseLearningRate
        self.slope = slope
        self.lowerBound = lowerBound
        self.startStep = startStep
    }

    @inlinable
    public func callAsFunction(forStep step: UInt64) -> Scalar {
        let learningRate = baseLearningRate(forStep: step)
        if step < startStep { return learningRate }
        let step = step - startStep
        let decayed = learningRate + Scalar(step) * slope
        return max(lowerBound * learningRate, decayed)
    }
}

/// Exponentially decayed learning rate.
///
/// The decayed learning rate is computed as follows:
/// ```swift
/// let initial = baseLearningRate(forStep: step)
/// let decay = decayRate ^ (step / decayStepCount)
/// let decayedLearningRate = initial * ((1 - lowerBound) * decay + lowerBound)
/// ```
/// where if `staircase = true`, then `step / decayStepCount` uses integer division and the decayed
/// learning rate follows a staircase function.
public struct ExponentiallyDecayedLearningRate<BaseLearningRate: LearningRate>: LearningRate
    where BaseLearningRate.Scalar: ElementaryFunctions {
    public typealias Scalar = BaseLearningRate.Scalar

    public let baseLearningRate: BaseLearningRate
    public let decayRate: Scalar
    public let decayStepCount: UInt64
    public let staircase: Bool
    public let lowerBound: Scalar
    public let startStep: UInt64

    /// Creates a new exponentially decayed learning rate.
    ///
    /// - Parameters:
    ///   - baseLearningRate: Learning rate to decay.
    ///   - decayRate: Decay rate.
    ///   - decayStepCount: Decay step count.
    ///   - staircase: If `true`, the decay will occur at discrete intervals.
    ///   - lowerBound: Minimum decayed learning rate value as a fraction of the original learning
    ///     rate value.
    ///   - startStep: Step after which to start decaying the learning rate.
    @inlinable
    public init(
        baseLearningRate: BaseLearningRate,
        decayRate: Scalar,
        decayStepCount: UInt64,
        staircase: Bool = false,
        lowerBound: Scalar = Scalar(0),
        startStep: UInt64 = 0
    ) {
        self.baseLearningRate = baseLearningRate
        self.decayRate = decayRate
        self.decayStepCount = decayStepCount
        self.staircase = staircase
        self.lowerBound = lowerBound
        self.startStep = startStep
    }

    @inlinable
    public func callAsFunction(forStep step: UInt64) -> Scalar {
        let learningRate = baseLearningRate(forStep: step)
        if step < startStep { return learningRate }
        let step = step - startStep
        let power = Scalar(step) / Scalar(decayStepCount)
        let decay = Scalar.pow(decayRate, staircase ? power.rounded(.down) : power)
        return learningRate * ((1 - lowerBound) * decay + lowerBound)
    }
}

/// Reciprocal square root decayed learning rate.
///
/// The decayed learning rate is computed as follows:
/// ```swift
/// let initial = baseLearningRate(forStep: step)
/// let decay = decayFactor / sqrt(max(step, decayThreshold))
/// let decayedLearningRate = initial * ((1 - lowerBound) * decay + lowerBound)
/// ```
public struct RSqrtLearningRateDecay<BaseLearningRate: LearningRate>: LearningRate
    where BaseLearningRate.Scalar: ElementaryFunctions {
    public typealias Scalar = BaseLearningRate.Scalar

    public let baseLearningRate: BaseLearningRate
    public let decayFactor: Scalar
    public let decayThreshold: Scalar
    public let lowerBound: Scalar
    public let startStep: UInt64

    /// Creates a new reciprocal square root decayed learning rate.
    ///
    /// - Parameters:
    ///   - baseLearningRate: Learning rate to decay.
    ///   - decayFactor: Decay factor.
    ///   - decayThreshold: Decay threshold.
    ///   - lowerBound: Minimum decayed learning rate value as a fraction of the original learning
    ///     rate value.
    ///   - startStep: Step after which to start decaying the learning rate.
    @inlinable
    public init(
        baseLearningRate: BaseLearningRate,
        decayFactor: Scalar,
        decayThreshold: Scalar,
        lowerBound: Scalar = Scalar(0),
        startStep: UInt64 = 0
    ) {
        self.baseLearningRate = baseLearningRate
        self.decayFactor = decayFactor
        self.decayThreshold = decayThreshold
        self.lowerBound = lowerBound
        self.startStep = startStep
    }

    @inlinable
    public func callAsFunction(forStep step: UInt64) -> Scalar {
        let learningRate = baseLearningRate(forStep: step)
        if step < startStep { return learningRate }
        let step = step - startStep
        let decay = decayFactor / Scalar.sqrt(max(Scalar(step), decayThreshold))
        return learningRate * ((1 - lowerBound) * decay + lowerBound)
    }
}

/// Cosine decayed learning rate.
///
/// The decayed learning rate is computed as follows:
/// ```swift
/// let initial = baseLearningRate(forStep: step)
/// let decay = 0.5 * (1 + cos(pi * min(step, cycleStepCount) / cycleStepCount))
/// let decayedLearningRate = initial * ((1 - lowerBound) * decay + lowerBound)
/// ```
public struct CosineDecayedLearningRate<BaseLearningRate: LearningRate>: LearningRate
    where BaseLearningRate.Scalar: ElementaryFunctions {
    public typealias Scalar = BaseLearningRate.Scalar

    public let baseLearningRate: BaseLearningRate
    public let cycleStepCount: UInt64
    public let lowerBound: Scalar
    public let startStep: UInt64

    /// Creates a new cosine decayed learning rate.
    ///
    /// - Parameters:
    ///   - baseLearningRate: Learning rate to decay.
    ///   - cycleStepCount: Cosine decay cycle in terms of number of steps.
    ///   - lowerBound: Minimum decayed learning rate value as a fraction of the original learning
    ///     rate value.
    ///   - startStep: Step after which to start decaying the learning rate.
    @inlinable
    public init(
        baseLearningRate: BaseLearningRate,
        cycleStepCount: UInt64,
        lowerBound: Scalar = Scalar(0),
        startStep: UInt64 = 0
    ) {
        self.baseLearningRate = baseLearningRate
        self.cycleStepCount = cycleStepCount
        self.lowerBound = lowerBound
        self.startStep = startStep
    }

    @inlinable
    public func callAsFunction(forStep step: UInt64) -> Scalar {
        let learningRate = baseLearningRate(forStep: step)
        if step < startStep { return learningRate }
        let step = step - startStep
        let cosine = Scalar.cos(Scalar(min(step, cycleStepCount)))
        let decay = (1 + cosine) * Scalar.pi / Scalar(2 * cycleStepCount)
        return learningRate * ((1 - lowerBound) * decay + lowerBound)
    }
}

/// Cycle-linear 10x decayed learning rate.
///
/// The decayed learning rate is computed as follows:
/// ```swift
/// let initial = baseLearningRate(forStep: step)
/// let cyclePosition = 1 - abs((step % (2 * cycleStepCount) - cycleStepCount) / cycleStepCount)
/// let decay = (0.1 + cyclePosition) * 3
/// let decayedLearningRate = initial * ((1 - lowerBound) * decay + lowerBound)
/// ```
public struct CycleLinear10xLearningRateDecay<BaseLearningRate: LearningRate>: LearningRate {
    public typealias Scalar = BaseLearningRate.Scalar

    public let baseLearningRate: BaseLearningRate
    public let cycleStepCount: UInt64
    public let lowerBound: Scalar
    public let startStep: UInt64

    /// Creates a new cycle-linear 10x decayed learning rate.
    ///
    /// - Parameters:
    ///   - baseLearningRate: Learning rate to decay.
    ///   - cycleStepCount: Cycle-linear 10x decay cycle in terms of number of steps.
    ///   - lowerBound: Minimum decayed learning rate value as a fraction of the original learning
    ///     rate value.
    ///   - startStep: Step after which to start decaying the learning rate.
    @inlinable
    public init(
        baseLearningRate: BaseLearningRate,
        cycleStepCount: UInt64,
        lowerBound: Scalar = Scalar(0),
        startStep: UInt64 = 0
    ) {
        self.baseLearningRate = baseLearningRate
        self.cycleStepCount = cycleStepCount
        self.lowerBound = lowerBound
        self.startStep = startStep
    }

    @inlinable
    public func callAsFunction(forStep step: UInt64) -> Scalar {
        let learningRate = baseLearningRate(forStep: step)
        if step < startStep { return learningRate }
        let step = step - startStep
        let ratio = Scalar((step % (2 * cycleStepCount) - cycleStepCount)) / Scalar(cycleStepCount)
        let cyclePosition = 1 - abs(ratio)
        let decay = (1 / Scalar(10) + cyclePosition) * 3 // 10x difference in each cycle (0.3 - 3).
        return learningRate * ((1 - lowerBound) * decay + lowerBound)
    }
}

/// Linearly warmed-up learning rate.
///
/// For the first `warmUpStepCount` steps the base learning rate is multiplied with:
/// ```
/// warmUpOffset + ((1 - warmUpOffset) / warmUpStepCount) * step
/// ```
///
/// - Source: [Attention is All You Need (Section 5.3)](https://arxiv.org/pdf/1706.03762.pdf).
public struct LinearlyWarmedUpLearningRate<BaseLearningRate: LearningRate>: LearningRate {
    public typealias Scalar = BaseLearningRate.Scalar

    public let baseLearningRate: BaseLearningRate
    public let warmUpStepCount: UInt64
    public let warmUpOffset: Scalar

    /// Creates a new linear learning rate warm-up schedule.
    ///
    /// - Parameters:
    ///   - baseLearningRate: Learning rate to warm-up.
    ///   - warmUpStepCount: Number of warm-up steps.
    ///   - warmUpOffset: Linear schedule offset.
    @inlinable
    public init(
        baseLearningRate: BaseLearningRate,
        warmUpStepCount: UInt64,
        warmUpOffset: Scalar
    ) {
        self.baseLearningRate = baseLearningRate
        self.warmUpStepCount = warmUpStepCount
        self.warmUpOffset = warmUpOffset
    }

    @inlinable
    public func callAsFunction(forStep step: UInt64) -> Scalar {
        let learningRate = baseLearningRate(forStep: step)
        if step >= warmUpStepCount { return learningRate }
        let factor = warmUpOffset + ((1 - warmUpOffset) / Scalar(warmUpStepCount)) * Scalar(step)
        return learningRate * factor
    }
}

/// Exponentially warmed-up learning rate.
///
/// For the first `warmUpStepCount` steps the base learning rate is multiplied with:
/// ```
/// exp(log(warmUpFactor) / step) ^ (warmUpStepCount - step)
/// ```
///
/// - Source: [Attention is All You Need (Section 5.3)](https://arxiv.org/pdf/1706.03762.pdf).
public struct ExponentialLearningRateWarmUp<BaseLearningRate: LearningRate>: LearningRate
    where BaseLearningRate.Scalar: ElementaryFunctions {
    public typealias Scalar = BaseLearningRate.Scalar

    public let baseLearningRate: BaseLearningRate
    public let warmUpStepCount: UInt64
    public let warmUpFactor: Scalar

    /// Creates a new linear learning rate warm-up schedule.
    ///
    /// - Parameters:
    ///   - baseLearningRate: Learning rate to warm-up.
    ///   - warmUpStepCount: Number of warm-up steps.
    ///   - warmUpFactor: Warm-up learning rate scaling factor.
    @inlinable
    public init(
        baseLearningRate: BaseLearningRate,
        warmUpStepCount: UInt64,
        warmUpFactor: Scalar
    ) {
        self.baseLearningRate = baseLearningRate
        self.warmUpStepCount = warmUpStepCount
        self.warmUpFactor = warmUpFactor
    }

    @inlinable
    public func callAsFunction(forStep step: UInt64) -> Scalar {
        let learningRate = baseLearningRate(forStep: step)
        if step >= warmUpStepCount { return learningRate }
        let base = Scalar.exp(Scalar.log(warmUpFactor) / Scalar(warmUpStepCount))
        let factor = Scalar.pow(base, Scalar(warmUpStepCount - step))
        return learningRate * factor
    }
}
