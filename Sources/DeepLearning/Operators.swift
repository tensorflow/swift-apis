// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#if !COMPILING_TENSORFLOW_MODULE
import TensorFlow
#endif

/// Returns the values of the specified tensor to the nearest integer, element-wise.
public func round<Scalar: BinaryFloatingPoint>(_ x: Tensor<Scalar>) -> Tensor<Scalar> {
    return Raw.round(x)
}

/// Returns a tensor with the same shape and scalars as specified tensor.
@differentiable
public func identity<Scalar>(_ x: Tensor<Scalar>) -> Tensor<Scalar> {
    return x
}
