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
import Foundation

/// A TensorFlow checkpoint file reader.
public class TensorFlowCheckpointReader {
    @usableFromInline internal let status: OpaquePointer
    @usableFromInline internal let handle: OpaquePointer

    /// URL of the checkpoint file.
    public let checkpointPath: URL

    /// Number of tensors stored in the checkpoint.
    public var tensorCount: Int { Int(TF_CheckpointReaderSize(handle)) }

    /// Names of the tensors stored in the checkpoint.
    public var tensorNames: [String] {
        (0..<tensorCount).map {
            String(cString: TF_CheckpointReaderGetVariable(handle, Int32($0)))
        }
    }

    /// Creates a new TensorFlow checkpoint reader.
    ///
    /// - Arguments:
    ///   - checkpointPath: URL of the checkpoint file.
    @inlinable
    public init?(checkpointPath: URL) {
        self.status = TF_NewStatus()
        self.handle = TF_NewCheckpointReader(checkpointPath.path, status)
        checkOk(status)
        self.checkpointPath = checkpointPath
    }

    deinit {
        TF_DeleteCheckpointReader(handle)
    }

    /// Returns `true` if the checkpoint contains a tensor with the provided name.
    @inlinable
    public func contains(tensorNamed name: String) -> Bool {
        TF_CheckpointReaderHasTensor(handle, name) > 0
    }

    /// Returns the shape of the tensor with the provided name stored in the checkpoint.
    @inlinable
    public func shape(ofTensorNamed name: String) -> TensorShape {
        let rank = TF_CheckpointReaderGetVariableNumDims(handle, name)
        let dimensions = UnsafeMutablePointer<Int64>.allocate(capacity: Int(rank))
        defer { dimensions.deallocate() }
        TF_CheckpointReaderGetVariableShape(handle, name, dimensions, rank, status)
        checkOk(status)
        let dimensionsBufferPointer = UnsafeBufferPointer(start: dimensions, count: Int(rank))
        return TensorShape([Int64](dimensionsBufferPointer).map(Int.init))
    }

    /// Returns the data type of the tensor with the provided name stored in the checkpoint.
    @inlinable
    public func dataType(ofTensorNamed name: String) -> TensorDataType {
        TensorDataType(TF_CheckpointReaderGetVariableDataType(handle, name))
    }

    /// Loads and returns the value of the tensor with the provided name stored in the checkpoint.
    @inlinable
    public func load<Scalar: _TensorFlowDataTypeCompatible>(
        tensorNamed name: String
    ) -> ShapedArray<Scalar> {
        let pointer = TF_CheckpointReaderGetTensor(handle, name, status)
        checkOk(status)
        return ShapedArray<Scalar>(owning: pointer!)
    }
}
