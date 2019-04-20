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

#if COMPILING_TENSORFLOW_MODULE
infix operator .==: ComparisonPrecedence
#endif

//===------------------------------------------------------------------------------------------===//
// Tensor Properties
//===------------------------------------------------------------------------------------------===//

public extension Tensor {
  /// The rank of the tensor, represented as a `Tensor<Int32>`.
  @inlinable
  var rankTensor: Tensor<Int32> {
    return Raw.rank(self)
  }

  /// The dimensions of the tensor, represented as a `Tensor<Int32>`.
  @inlinable
  var shapeTensor: Tensor<Int32> {
    return Raw.shape(self)
  }

  /// The number of scalars in the tensor, represented as a `Tensor<Int32>`.
  @inlinable
  var scalarCountTensor: Tensor<Int32> {
    return Raw.size(self)
  }
}

//===------------------------------------------------------------------------------------------===//
// Description and Visualization
//===------------------------------------------------------------------------------------------===//

// String conversion.
extension Tensor: CustomStringConvertible {
    /// A textual representation of the tensor.
    ///
    /// - Note: use `fullDescription` for a non-pretty-printed description showing all scalars.
    public var description: String {
        return array.description
    }
}

public extension Tensor {
    /// A textual representation of the tensor. Returns a summarized description if `summarize` is
    /// true and the element count exceeds twice the `edgeElementCount`.
    ///
    /// - Parameters:
    ///   - lineWidth: The max line width for printing. Used to determine number of scalars to print
    ///     per line.
    ///   - edgeElementCount: The maximum number of elements to print before and after summarization
    ///     via ellipses (`...`).
    ///   - summarizing: If true, summarize description if element count exceeds twice
    ///     `edgeElementCount`.
    func description(
        lineWidth: Int = 80,
        edgeElementCount: Int = 3,
        summarizing: Bool = false
    ) -> String {
        return array.description(
            lineWidth: lineWidth,
            edgeElementCount: edgeElementCount,
            summarizing: summarizing)
    }

    /// A full, non-pretty-printed textual representation of the tensor, showing
    /// all scalars.
    var fullDescription: String {
        return array.fullDescription
    }
}

// Xcode Playground display conversion.
extension Tensor: CustomPlaygroundDisplayConvertible {
    public var playgroundDescription: Any {
        return description
    }
}

// Mirror representation, used by debugger/REPL.
extension Tensor: CustomReflectable {
    public var customMirror: Mirror {
        return Mirror(self, children: [], displayStyle: .struct)
    }
}

//===------------------------------------------------------------------------------------------===//
// Codable Conformance
//===------------------------------------------------------------------------------------------===//

extension Tensor: Codable where Scalar: Codable {
    @inlinable
    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(array)
    }

    @inlinable
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let array = try container.decode(ShapedArray<Scalar>.self)
        self.init(array)
    }
}
