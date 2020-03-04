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

/// A numerical optimizer.
///
/// Optimizers apply an optimization algorithm to update a differentiable model.
public protocol Optimizer: CopyableToDevice {
  /// The type of the model to optimize.
  associatedtype Model: Differentiable
  /// The scalar parameter type.
  associatedtype Scalar: FloatingPoint
  /// The learning rate.
  var learningRate: Scalar { get set }
  /// Updates the given model along the given direction.
  mutating func update(_ model: inout Model, along direction: Model.TangentVector)
}
