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

#if !COMPILING_TENSORFLOW_STDLIB_MODULE
import Tensor
#endif

public typealias ParameterInitializer<Scalar: TensorFlowScalar> = (TensorShape) -> Tensor<Scalar>

/// Returns a function that creates a tensor by initializing all its values to zeros.
public func zeros<Scalar: TensorFlowFloatingPoint>() -> ParameterInitializer<Scalar> {
    { Tensor(zeros: $0) }
}

/// Returns a function that creates a tensor by initializing all its values to the provided value.
public func constantInitializer<Scalar: TensorFlowFloatingPoint>(
    value: Scalar
) -> ParameterInitializer<Scalar> {
    { Tensor(repeating: value, shape: $0) }
}

/// Returns a function that creates a tensor by initializing it to the provided value. Note that
/// broadcasting of the provided value is *not* supported.
public func constantInitializer<Scalar: TensorFlowFloatingPoint>(
    value: Tensor<Scalar>
) -> ParameterInitializer<Scalar> {
    {
        precondition(
            value.shape == $0,
            "The constant tensor shape (\(value.shape)) must match the requested shape \($0).")
        return value
    }
}

/// Returns a function that creates a tensor by performing Glorot uniform initialization for the 
/// specified shape, randomly sampling scalar values from a uniform distribution between `-limit` 
/// and `limit`, generated by the default random number generator, where limit is
/// `sqrt(6 / (fanIn + fanOut))`, and `fanIn`/`fanOut` represent the number of input and output
/// features multiplied by the receptive field, if present.
public func glorotUniform<Scalar: TensorFlowFloatingPoint>(
    seed: TensorFlowSeed = Context.local.randomSeed
) -> ParameterInitializer<Scalar> {
    { Tensor<Scalar>(glorotUniform: $0, seed: seed) }
}

/// Returns a function that creates a tensor by initializing all its values randomly from a
/// truncated Normal distribution. The generated values follow a Normal distribution with mean
/// `mean` and standard deviation `standardDeviation`, except that values whose magnitude is more
/// than two standard deviations from the mean are dropped and resampled.
///
/// - Parameters:
///   - mean: Mean of the Normal distribution.
///   - standardDeviation: Standard deviation of the Normal distribution.
///
///- Returns: A truncated normal parameter initializer function.
public func truncatedNormalInitializer<Scalar: TensorFlowFloatingPoint>(
    mean: Tensor<Scalar> = Tensor<Scalar>(0),
    standardDeviation: Tensor<Scalar> = Tensor<Scalar>(1),
    seed: TensorFlowSeed = Context.local.randomSeed
) -> ParameterInitializer<Scalar> {
    {
        Tensor<Scalar>(
            randomTruncatedNormal: $0,
            mean: mean,
            standardDeviation: standardDeviation,
            seed: seed)
    }
}
