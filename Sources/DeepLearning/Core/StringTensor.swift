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

import CTensorFlow
import TensorFlowCore

//===------------------------------------------------------------------------------------------===//
// StringTensor
//===------------------------------------------------------------------------------------------===//

/// `StringTensor` is a multi-dimensional array whose elements are `String`s.
@_fixed_layout
public struct StringTensor {
    /// The underlying `TensorHandle`.
    /// - Note: `handle` is public to allow user defined ops, but should not normally be used
    ///   otherwise.
    public let handle: TensorHandle<String>

    @inlinable
    public init(handle: TensorHandle<String>) {
        self.handle = handle
    }
}

//===------------------------------------------------------------------------------------------===//
// Initialization
//===------------------------------------------------------------------------------------------===//

public extension StringTensor {
    /// Creates a tensor from a scalar value.
    @inlinable @inline(__always)
    init(_ value: String) {
        self.init(handle: _TFStringTensorFromString(value))
    }

    /// Creates a 1D tensor from contiguous scalars.
    ///
    /// - Parameters:
    ///   - vector: The scalar contents of the tensor.
    @inlinable @inline(__always)
    init(_ vector: [String]) {
        self.init(handle: _TFStringTensorFromStrings1D(vector))
    }
}

//===------------------------------------------------------------------------------------------===//
// Array Conversion
//===------------------------------------------------------------------------------------------===//

public extension StringTensor {
    var array: ShapedArray<String> {
        debugLog("Returning a host copy of string array.")
        return handle.makeHostCopy()
    }

    var scalars: [String] {
        return array.scalars
    }
}
