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

/// A pullback function that performs the transpose of broadcasting two `Tensors`.
public struct BroadcastingPullback {
  /// Constructs the pullback from broadcasting `lhs` and `rhs`.
  public init<T: TensorFlowFloatingPoint, U: TensorFlowFloatingPoint>(
    _ lhs: Tensor<T>, _ rhs: Tensor<U>
  ) {
    lhsShape = lhs.shape.dimensions.map { Int64($0) }
    rhsShape = rhs.shape.dimensions.map { Int64($0) }
  }

  public func callAsFunction<T: TensorFlowFloatingPoint, U: TensorFlowFloatingPoint>(
    _ lhsGrad: Tensor<T>, _ rhsGrad: Tensor<U>
  ) -> (Tensor<T>, Tensor<U>) {
    if lhsShape == rhsShape { return (lhsGrad, rhsGrad) }
    let (lhsAxes, rhsAxes) = BroadcastingPullback.computeReductionAxes(lhsShape, rhsShape)
    return (
      _Raw.reshape(_Raw.sum(lhsGrad, reductionIndices: lhsAxes, keepDims: false), shape: lhsShape),
      _Raw.reshape(_Raw.sum(rhsGrad, reductionIndices: rhsAxes, keepDims: false), shape: rhsShape)
    )
  }

  /// Compute the axis needed to sum along in order to map back from the
  /// broadcasted shape to the individual argument shapes.
  @usableFromInline
  static func computeReductionAxes(
    _ lhsShape: [Int64], _ rhsShape: [Int64]
  ) -> (lhsAxes: [Int64], rhsAxes: [Int64]) {
    var shape0 = lhsShape
    var shape1 = rhsShape
    var reduceIdx0 = [Int64]()
    var reduceIdx1 = [Int64]()
    shape0.reverse()
    shape1.reverse()
    while shape0.count < shape1.count { shape0.append(1) }
    while shape1.count < shape0.count { shape1.append(1) }
    let n = shape1.count
    for i in 0..<n {
      let d0 = shape0[i]
      let d1 = shape1[i]
      if d0 == 1 { reduceIdx0.append(Int64(n - i - 1)) }
      if d1 == 1 { reduceIdx1.append(Int64(n - i - 1)) }
    }
    reduceIdx0.reverse()
    reduceIdx1.reverse()
    return (lhsAxes: reduceIdx0, rhsAxes: reduceIdx1)
  }

  let lhsShape: [Int64]
  let rhsShape: [Int64]
}
