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

//===------------------------------------------------------------------------------------------===//
// StringTensor
//===------------------------------------------------------------------------------------------===//

/// This compiler builtin is known by the partitioning pass, which recognizes it
/// and promotes calls to it to being in graph when it can. This signature was
/// designed to align with the requirements of the `Const` TensorFlow operation.
@inlinable @inline(never)
@_silgen_name("__tf_string_tensor_from_strings")
public func _TFStringTensorFromStrings(_ scalars: [String], shape: [Int]) -> TensorHandle<String> {
    let contiguousSize = shape.reduce(1, *)
    precondition(scalars.count == contiguousSize, "The number of scalars does not match the shape.")

    // utf8CString is null-terminated. TF APIs want the strings without null-terminators.
    let cStrings = scalars.map { $0.utf8CString.dropLast() }

    let tfEncodedSizes = cStrings.map { TF_StringEncodedSize($0.count) }

    // Format information copied from tensorflow/c/c_api.h:
    // The format for TF_STRING tensors is:
    //   start_offset: array[uint64]
    //   data:         byte[...]
    //
    //   The string length (as a varint), followed by the contents of the string
    //   is encoded at data[start_offset[i]]].

    // The size of the "start_offset" region.
    let startOffsetsByteCount = scalars.count * MemoryLayout<UInt64>.stride

    // The size of the "data" region.
    let dataByteCount = tfEncodedSizes.reduce(0, +) * MemoryLayout<UInt8>.stride

    return TensorHandle(
        shape: shape,
        byteCount: startOffsetsByteCount + dataByteCount,
        bufferInitializer: { tensorBuffer in
            // Initialize the "start_offset" region.
            var startOffset: UInt64 = 0
            var startOffsetAddr = tensorBuffer.bindMemory(to: UInt64.self, capacity: scalars.count)
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
                        buffer.baseAddress,
                        buffer.count,
                        dataAddr,
                        tfEncodedSize,
                        status)
                }
                checkOk(status)
                dataAddr = dataAddr.advanced(by: tfEncodedSize)
            }
            TF_DeleteStatus(status)
        })
}

@inlinable @inline(never)
@_silgen_name("__tf_string_tensor_from_string")
public func _TFStringTensorFromString(_ scalar: String) -> TensorHandle<String> {
    return _TFStringTensorFromStrings([scalar], shape: [])
}

@inlinable @inline(never)
@_silgen_name("__tf_string_tensor_from_strings_1d")
public func _TFStringTensorFromStrings1D(_ scalars: [String]) -> TensorHandle<String> {
    return _TFStringTensorFromStrings(scalars, shape: [scalars.count])
}
