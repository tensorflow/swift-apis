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

// NOTE: it may be possible to edit `TensorShape` to support "labeled tensors". Dimensions may be
// either an Int or an enum representing a label.

/// A struct representing the shape of a tensor.
///
/// `TensorShape` is a thin wrapper around an array of integers that represent shape dimensions. All
/// tensor types use `TensorShape` to represent their shape.
@frozen
public struct TensorShape: ExpressibleByArrayLiteral {
    /// The dimensions of the shape.
    public var dimensions: [Int]

    /// Initialize with an array of dimensions. The rank of the tensor is the length of the array.
    /// - Parameter dimensions: The shape dimensions.
    @inlinable
    public init(_ dimensions: [Int]) {
        self.dimensions = dimensions
    }

    /// Initialize with a collection of dimensions. The rank of the tensor is the length of the
    /// collection.
    /// - Parameter dimensions: The shape dimensions.
    @inlinable
    public init<C: Collection>(_ dimensions: C) where C.Element == Int {
        self.dimensions = Array(dimensions)
    }

    /// Initialize with an array literal representing the shape dimensions. The rank of the tensor
    /// is the number of dimensions.
    /// - Parameter dimensions: The shape dimensions.
    @inlinable
    public init(arrayLiteral elements: Int...) {
        self.init(elements)
    }

    /// Initialize with variadic elements representing the shape dimensions. The rank of the tensor
    /// is the number of elements.
    /// - Parameter dimensions: The shape dimensions.
    @inlinable
    public init(_ elements: Int...) {
        self.init(elements)
    }

    @inlinable
    public init(repeating repeatedValue: Int, count: Int) {
        self.init(Array(repeating: repeatedValue, count: count))
    }

    /// The rank of the shape (i.e. the number of dimensions).
    @inlinable
    public var rank: Int {
        return dimensions.count
    }

    /// The size of the shape as a contiguously stored array.
    @inlinable
    public var contiguousSize: Int {
        return dimensions.reduce(1, *)
    }
}

public extension TensorShape {
    /// The rank of the shape (i.e. the number of dimensions).
    @inlinable
    var count: Int {
        return dimensions.count
    }

    @inlinable
    var indices: Range<Int> {
        return dimensions.indices.lowerBound ..< dimensions.indices.upperBound
    }

    @inlinable
    var startIndex: Int {
        return dimensions.startIndex
    }

    @inlinable
    var endIndex: Int {
        return dimensions.endIndex
    }

    /// Access the size of the i-th dimension.
    /// - Parameter index: The index of a dimension.
    @inlinable
    subscript(index: Int) -> Int {
        _read { yield dimensions[index] }
        _modify { yield &dimensions[index] }
    }

    /// Access the size of the i-th dimension.
    /// - Parameter index: The index of a dimension.
    @inlinable
    subscript(bounds: Range<Int>) -> TensorShape {
        get { return TensorShape(dimensions[bounds]) }
        set { dimensions[bounds] = ArraySlice(newValue.dimensions) }
    }
}

extension TensorShape: Equatable {
    @inlinable
    public static func == (lhs: TensorShape, rhs: TensorShape) -> Bool {
        return lhs.dimensions == rhs.dimensions
    }
}

extension TensorShape: Codable {
    @inlinable
    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(dimensions)
    }

    @inlinable
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let dimensions = try container.decode([Int].self)
        self.init(dimensions)
    }
}

extension TensorShape: CustomStringConvertible {
    public var description: String {
      return dimensions.description
    }
}
