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

/// A TensorFlow checkpoint file reader.
@available(*, deprecated, message: """
  TensorFlowCheckpointReader will be removed in S4TF v0.11. Please use CheckpointReader from swift-models
  (https://github.com/tensorflow/swift-models/blob/master/Support/Checkpoints/CheckpointReader.swift)
  instead.
  """)
public class TensorFlowCheckpointReader {
  internal let status: OpaquePointer
  internal let handle: OpaquePointer

  /// The path to the checkpoint file.
  public let checkpointPath: String

  /// The number of tensors stored in the checkpoint.
  public var tensorCount: Int { Int(TF_CheckpointReaderSize(handle)) }

  /// The names of the tensors stored in the checkpoint.
  public var tensorNames: [String] {
    (0..<tensorCount).map {
      String(cString: TF_CheckpointReaderGetVariable(handle, Int32($0)))
    }
  }

  /// Creates a new TensorFlow checkpoint reader.
  ///
  /// - Arguments:
  ///   - checkpointPath: Path to the checkpoint file.
  public init(checkpointPath: String) {
    self.status = TF_NewStatus()
    self.handle = TF_NewCheckpointReader(checkpointPath, status)
    checkOk(status)
    self.checkpointPath = checkpointPath
  }

  deinit {
    TF_DeleteStatus(status)
    TF_DeleteCheckpointReader(handle)
  }

  /// Returns `true` if the checkpoint contains a tensor with the provided name.
  public func containsTensor(named name: String) -> Bool {
    TF_CheckpointReaderHasTensor(handle, name) > 0
  }

  /// Returns the shape of the tensor with the provided name stored in the checkpoint.
  public func shapeOfTensor(named name: String) -> TensorShape {
    let rank = TF_CheckpointReaderGetVariableNumDims(handle, name)
    let dimensions = UnsafeMutablePointer<Int64>.allocate(capacity: Int(rank))
    defer { dimensions.deallocate() }
    TF_CheckpointReaderGetVariableShape(handle, name, dimensions, rank, status)
    checkOk(status)
    let dimensionsBufferPointer = UnsafeBufferPointer(start: dimensions, count: Int(rank))
    return TensorShape([Int64](dimensionsBufferPointer).map(Int.init))
  }

  /// Returns the scalar type of the tensor with the provided name stored in the checkpoint.
  public func scalarTypeOfTensor(named name: String) -> Any.Type {
    let dataType = TensorDataType(TF_CheckpointReaderGetVariableDataType(handle, name))
    switch dataType._cDataType {
    case TF_BOOL: return Bool.self
    case TF_INT8: return Int8.self
    case TF_UINT8: return UInt8.self
    case TF_INT16: return Int16.self
    case TF_UINT16: return UInt16.self
    case TF_INT32: return Int32.self
    case TF_UINT32: return UInt32.self
    case TF_INT64: return Int64.self
    case TF_UINT64: return UInt64.self
    case TF_BFLOAT16: return BFloat16.self
    case TF_FLOAT: return Float.self
    case TF_DOUBLE: return Double.self
    case TF_STRING: return String.self
    default: fatalError("Unhandled type: \(dataType)")
    }
  }

  /// Loads and returns the value of the tensor with the provided name stored in the checkpoint.
  public func loadTensor<Scalar: _TensorFlowDataTypeCompatible>(
    named name: String
  ) -> ShapedArray<Scalar> {
    let pointer = TF_CheckpointReaderGetTensor(handle, name, status)
    checkOk(status)
    return ShapedArray<Scalar>(owning: pointer!)
  }
}
