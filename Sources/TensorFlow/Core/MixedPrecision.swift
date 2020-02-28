// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

extension Tensor {
  /// Returns true if the physical scalar type is reduced precision.
  ///
  /// Currently, reduced precision physical scalar types include only `BFloat16`.
  public var isReducedPrecision: Bool {
    // TODO: Implement.
    return false
  }

  /// Promotes a scalar to a tensor with the same device and precision as the given tensor.
  @differentiable( where Scalar: TensorFlowFloatingPoint)
  public init(_ value: Scalar, deviceAndPrecisionLike tensor: Tensor) {
    let tmp = Tensor(value, on: tensor.device)
    self = tensor.isReducedPrecision ? tmp.toReducedPrecision : tmp
  }
}

extension Tensor {
  /// Returns a copy of `self` converted to `BFloat16` physical scalar type.
  public var toReducedPrecision: Self {
    // TODO: Implement.
    self
  }

  /// Returns a copy of `self` converted to `Scalar` physical scalar type.
  public var toFullPrecision: Self {
    // TODO: Implement.
    self
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @derivative(of: toReducedPrecision)
  func _vjpToReducedPrecision() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (toReducedPrecision, { $0.toFullPrecision })
  }

  @derivative(of: toFullPrecision)
  func _vjpToFullPrecision() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (toFullPrecision, { $0.toReducedPrecision })
  }
}
