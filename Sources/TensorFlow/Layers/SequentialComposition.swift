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

/// The sequential composition of the elements of some base collection of
/// layers.
///
/// The first element of the base collection is applied first by the composite,
/// so it is the inner call of the composition or the last in the sequence of
/// composed layers in “f ∘ g” notation.
public struct SequentialComposition<Base: Collection>
  where Base.Element : Layer, Base.Element.Input == Base.Element.Output
{
  /// The layers to be composed.
  private let base: Base

  /// Creates an instance that composes the elenents of `base` in order, such
  /// that the first element of `base` is applied first by the composite.
  public init(_ base: Base) {
    self.base = base
  }
  
  /// Performs the composed layer application.
  @differentiable
  public func callAsFunction(
    _ input: Base.Element.Input
  ) -> Base.Element.Output {
    var r = input
    for l in base { r = l(r) }
    return r
  }

  // TODO: How is this documented?
  public init(@LayerBuilder layers: () -> Self) {
    self = layers()
  }
}

extension Collection where Element : Layer, Element.Input == Element.Output
{
  /// A sequential composition of the elements of `self`.
  ///
  /// The first element of `self` is applied first by the composite, i.e. it is
  /// the inner call of the composition or the last in the sequence of composed
  /// functions in “f ∘ g” notation.
  var sequentiallyComposed: SequentialComposition<Self> { .init(self) }
}
