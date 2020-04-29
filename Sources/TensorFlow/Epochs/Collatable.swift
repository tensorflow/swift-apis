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

/// Types whose elements can be collated in some higher-rank element of the same
/// type (example: tensors, tuple of tensors)
public protocol Collatable {
  init<BatchSamples: Collection>(collating: BatchSamples)
  where BatchSamples.Element == Self
}

// Tensor are collated using stacking
extension Tensor: Collatable {
  public init<BatchSamples: Collection>(collating samples: BatchSamples)
  where BatchSamples.Element == Self {
    let batchSamples = samples.indices.concurrentMap { samples[$0] }
    self.init(stacking: batchSamples)
  }
}

// TODO: derived conformance

extension Collection where Element: Collatable {
  /// The result of collating the elements of `self`.
  public var collated: Element { .init(collating: self) }

  /// Returns the elements of `self`, padded to maximal shape with `padValue`
  /// and collated.
  /// 
  /// - Parameter atStart: adds the padding at the beginning if this is `true`
  ///   and the end otherwise. The default value is `false`.
  public func paddedAndCollated<Scalar: Numeric>(
    with padValue: Scalar, atStart: Bool = false
  ) -> Element
  where Element == Tensor<Scalar> {
    let firstShape = self.first!.shapeTensor
    let otherShapes = self.dropFirst().lazy.map(\.shapeTensor)
    let paddedShape = otherShapes.reduce(firstShape) { TensorFlow.max($0, $1) }
      .scalars.lazy.map { Int($0) }

    let r = self.lazy.map { t in
      t.padded(
        forSizes: zip(t.shape, paddedShape).map {
          (before: atStart ? $1 - $0 : 0, after: atStart ? 0 : $1 - $0)
        },
        with: padValue)
    }
    return r.collated
  }
}
