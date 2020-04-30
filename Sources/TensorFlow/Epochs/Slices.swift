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

/// A collection of the longest non-overlapping contiguous slices of some `Base`
/// collection, starting with its first element, and having some fixed maximum
/// length.
///
/// The elements of this collection, except for the last, all have a `count` of
/// `batchSize`, unless `Base.count % batchSize !=0`, in which case
/// the last batch's `count` is `base.count % batchSize.`
public struct Slices<Base: Collection> {
  /// The collection from which slices will be drawn.
  private let base: Base

  /// The maximum length of the slices.
  private let batchSize: Int

  /// Creates an instance that divides `base` into batches
  /// of size `n`.
  ///
  /// If `base.count % n != 0` the `count` of the last batch
  /// will be `base.count % n`
  public init(_ base: Base, batchSize n: Int) {
    self.base = base
    batchSize = n
  }
}

extension Slices: Collection {
  /// A position in `Slices`.
  public struct Index: Comparable {
    /// The range of base indices covered by the element at this position.
    var focus: Range<Base.Index>

    /// Returns true iff `l` precedes `r` in the collection.
    public static func < (l: Index, r: Index) -> Bool {
      l.focus.lowerBound < r.focus.lowerBound
    }
  }

  /// Returns the element at `i`.
  public subscript(i: Index) -> Base.SubSequence { base[i.focus] }

  /// Returns the base index that marks the end of the element of `self` that
  /// begins at `i` in the `base`, or `base.endIndex` if `i == base.endIndex`.
  private func sliceBoundary(after i: Base.Index) -> Base.Index {
    base.index(i, offsetBy: batchSize, limitedBy: base.endIndex)
      ?? base.endIndex
  }

  /// Returns the index after `i`.
  public func index(after i: Index) -> Index {
    Index(focus: i.focus.upperBound..<sliceBoundary(after: i.focus.upperBound))
  }

  /// Returns the first position in `self`.
  public var startIndex: Index {
    Index(focus: base.startIndex..<sliceBoundary(after: base.startIndex))
  }

  /// Returns the position one past the last element of `self`.
  public var endIndex: Index {
    Index(focus: base.endIndex..<base.endIndex)
  }
}

extension Collection {
  /// Returns the longest non-overlapping slices of `self`, starting with its
  /// first element, having a maximum length of `batchSize`.
  public func inBatches(of batchSize: Int) -> Slices<Self> {
    Slices(self, batchSize: batchSize)
  }
}
