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

/// A lazy selection of elements, in a given order, from some base collection.
public struct Sampling<Base: Collection, Selection: Collection>
where Selection.Element == Base.Index {
  /// The order that base elements appear in `self`.
  private let selection: Selection
  /// The base collection.
  private let base: Base

  /// Creates an instance from `base` and `selection`.
  public init(base: Base, selection: Selection) {
    self.selection = selection
    self.base = base
  }
}

extension Sampling: Collection {
  public typealias Element = Base.Element

  /// A type whose instances represent positions in `self`.
  public typealias Index = Selection.Index

  /// The position of the first element.
  public var startIndex: Index { selection.startIndex }

  /// The position one past the last element.
  public var endIndex: Index { selection.endIndex }

  /// Returns the element at `i`.
  public subscript(i: Index) -> Element { base[selection[i]] }

  /// Returns the position after `i`.
  public func index(after i: Index) -> Index { selection.index(after: i) }

  /// Returns the number of forward steps required to convert `start` into `end`.
  ///
  /// A negative result indicates that `end < start`.
  public func distance(from start: Index, to end: Index) -> Int {
    selection.distance(from: start, to: end)
  }

  /// Returns the position `n` places from `i`.
  public func index(_ i: Index, offsetBy n: Int) -> Index {
    selection.index(i, offsetBy: n)
  }

  /// Returns `i` offset by `distance` unless that requires passing `limit`, in
  /// which case `nil` is returned.
  public func index(
    _ i: Index, offsetBy distance: Int, limitedBy limit: Index
  ) -> Index? {
    selection.index(i, offsetBy: distance, limitedBy: limit)
  }
}

extension Sampling: BidirectionalCollection
  where Selection: BidirectionalCollection
{
  /// Returns the position before `i`.
  public func index(before i: Index) -> Index {
    selection.index(before: i)
  }
}

extension Sampling: RandomAccessCollection
  where Selection: RandomAccessCollection {}

extension Collection {
  /// Returns a collection of elements of `self` at the positions and in the order
  /// specified by `selection` without reading the elements of either collection.
  ///
  /// - Complexity: O(1)
  public func sampled<Selection: Collection>(at selection: Selection)
    -> Sampling<Self, Selection>
  {
    .init(base: self, selection: selection)
  }
}
