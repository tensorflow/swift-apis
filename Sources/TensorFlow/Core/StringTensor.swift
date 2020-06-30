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

import CTensorFlow

//===------------------------------------------------------------------------------------------===//
// StringTensor
//===------------------------------------------------------------------------------------------===//

/// `StringTensor` is a multi-dimensional array whose elements are `String`s.
@frozen
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

extension StringTensor {
  @inlinable
  public init(shape: TensorShape, scalars: [String]) {
    let contiguousSize = shape.contiguousSize
    precondition(
      scalars.count == contiguousSize,
      "The number of scalars does not match the shape.")

    // utf8CString is null-terminated. TF APIs want the strings without null-terminators.
    let cStrings = scalars.map { $0.utf8CString.dropLast() }

    // Note: `TENSORFLOW_MASTER` changes below are necessary for the new TensorFlow ABI-stable
    // unified string tensor design.
    #if TENSORFLOW_MASTER
      let byteCount = scalars.count * MemoryLayout<TF_TString>.stride

      let handle = TensorHandle<String>(
        shape: shape.dimensions,
        byteCount: byteCount,
        bufferInitializer: { tensorBuffer in
          var dataAddr =
            tensorBuffer.bindMemory(to: TF_TString.self, capacity: scalars.count)
          for cString in cStrings {
            TF_TString_Init(dataAddr)
            cString.withUnsafeBufferPointer { buffer in
              TF_TString_Copy(dataAddr, buffer.baseAddress, buffer.count)
            }
            dataAddr = dataAddr.advanced(by: 1)
          }
        })
      self.init(handle: handle)
    #else
      let tfEncodedSizes = cStrings.map { TF_StringEncodedSize($0.count) }

      // Format information copied from tensorflow/c/c_api.h:
      // The format for TF_STRING tensors is:
      //   start_offset: array[uint64]
      //   data:         byte[...]
      //
      //   The string length (as a varint), followed by the contents of the string is encoded at
      //   data[start_offset[i]]].
      // The size of the "start_offset" region.
      let startOffsetsByteCount = scalars.count * MemoryLayout<UInt64>.stride

      // The size of the "data" region.
      let dataByteCount = tfEncodedSizes.reduce(0, +) * MemoryLayout<UInt8>.stride

      let handle = TensorHandle<String>(
        shape: shape.dimensions,
        byteCount: startOffsetsByteCount + dataByteCount,
        bufferInitializer: { tensorBuffer in
          // Initialize the "start_offset" region.
          var startOffset: UInt64 = 0
          var startOffsetAddr =
            tensorBuffer.bindMemory(to: UInt64.self, capacity: scalars.count)
          for tfEncodedSize in tfEncodedSizes {
            startOffsetAddr.initialize(to: startOffset)
            startOffsetAddr = startOffsetAddr.advanced(by: 1)
            startOffset = startOffset + UInt64(tfEncodedSize)
          }

          // Initialize the "data" region.
          var dataAddr = tensorBuffer.advanced(by: startOffsetsByteCount)
            .bindMemory(to: Int8.self, capacity: dataByteCount)
          let status = TF_NewStatus()
          for (cString, tfEncodedSize) in zip(cStrings, tfEncodedSizes) {
            _ = cString.withUnsafeBufferPointer { buffer in
              TF_StringEncode(
                buffer.baseAddress, buffer.count, dataAddr, tfEncodedSize, status)
            }
            checkOk(status)
            dataAddr = dataAddr.advanced(by: tfEncodedSize)
          }
          TF_DeleteStatus(status)
        })
      self.init(handle: handle)
    #endif
  }

  /// Creates a 0-D `StringTensor` from a scalar value.
  @inlinable
  public init(_ value: String) {
    self.init(shape: [], scalars: [value])
  }

  /// Creates a 1-D `StringTensor` in from contiguous scalars.
  @inlinable
  public init(_ scalars: [String]) {
    self.init(shape: [scalars.count], scalars: scalars)
  }
}

//===------------------------------------------------------------------------------------------===//
// Array Conversion
//===------------------------------------------------------------------------------------------===//

extension StringTensor {
  public var array: ShapedArray<String> {
    debugLog("Returning a host copy of string array.")
    return handle.makeHostCopy()
  }

  public var scalars: [String] {
    return array.scalars
  }
}

//===------------------------------------------------------------------------------------------===//
// Element-wise comparison.
//===------------------------------------------------------------------------------------------===//
extension StringTensor {
  /// Computes `self == other` element-wise.
  /// - Note: `elementsEqual` supports broadcasting.
  @inlinable
  public func elementsEqual(_ other: StringTensor) -> Tensor<Bool> {
    return _RawTFEager.equal(self, other)
  }
}
