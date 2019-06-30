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

public typealias ParameterInitializer<Scalar: TensorFlowScalar> = (TensorShape) -> Tensor<Scalar>

/// Returns a function that creates a tensor by initializing all its values to zeros.
public func zeros<Scalar: TensorFlowFloatingPoint>() -> ParameterInitializer<Scalar> {
    { Tensor(zeros: $0) }
}

/// Returns a function that creates a tensor by performing Glorot uniform initialization for the 
/// specified shape, randomly sampling scalar values from a uniform distribution between `-limit` 
/// and `limit`, generated by the default random number generator, where limit is
/// `sqrt(6 / (fanIn + fanOut))`, and `fanIn`/`fanOut` represent the number of input and output
/// features multiplied by the receptive field, if present.
public func glorotUniform<Scalar: TensorFlowFloatingPoint>(
    seed: (Int32, Int32) = randomSeed()
) -> ParameterInitializer<Scalar> {
    { Tensor<Scalar>(glorotUniform: $0, seed: seed) }
}
