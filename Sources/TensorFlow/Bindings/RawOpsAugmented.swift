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

// Augment the `_Raw` interface with ops that take Swift integers for the
// shape attributes rather than requiring that they be passed as `Int32` tensors.
// This is useful for implementations that do not require the parameters be
// passed as Tensors.
extension _Raw {
  public static func argMax<
    T: TensorFlowNumeric,
    OutputType: TensorFlowIndex
  >(
    _ input: Tensor<T>,
    dimension: Int64
  ) -> Tensor<OutputType> {
    argMax(input, dimension: Tensor<Int32>(Int32(dimension)))
  }

  public static func mean<
    T: TensorFlowNumeric
  >(
    _ input: Tensor<T>,
    reductionIndices: [Int64],
    keepDims: Bool = false
  ) -> Tensor<T> {
    mean(
      input, reductionIndices: Tensor<Int32>(reductionIndices.map { Int32($0) }),
      keepDims: keepDims)
  }

  public static func reshape<
    T: TensorFlowScalar
  >(
    _ tensor: Tensor<T>,
    shape: [Int64]
  ) -> Tensor<T> {
    reshape(tensor, shape: Tensor<Int32>(shape.map { Int32($0) }))
  }

  public static func sum<
    T: TensorFlowNumeric
  >(
    _ input: Tensor<T>,
    reductionIndices: [Int64],
    keepDims: Bool = false
  ) -> Tensor<T> {
    sum(
      input, reductionIndices: Tensor<Int32>(reductionIndices.map { Int32($0) }),
      keepDims: keepDims)
  }

  public static func broadcastTo<
    T: TensorFlowScalar
  >(
    _ input: Tensor<T>,
    shape: [Int64]
  ) -> Tensor<T> {
    broadcastTo(input, shape: Tensor<Int32>(shape.map { Int32($0) }))
  }
}
