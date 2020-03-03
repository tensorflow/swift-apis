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

extension LazyTensorOperation {
  /// Returns a newly created TFE_Op with only the attributes set. NOTE: the
  /// caller should explicitly call `TFE_DeleteOp(tfeOp.op)` and
  /// `TFE_DeleteStatus(tfeOp.status)` to free the resources allocated in the
  /// newly created TFE_Op.
  private var tfeOp: TFE_Op {
    let op = TFE_Op(name, outputCount)
    for (name, value) in attributes {
      switch value {
      case .boolValue(let v): op.updateAttribute(name, v)
      case .intValue(let v): op.updateAttribute(name, v)
      case .floatValue(let v): op.updateAttribute(name, v)
      case .doubleValue(let v): op.updateAttribute(name, v)
      case .stringValue(let v): op.updateAttribute(name, v)
      case .boolArray(let v): op.updateAttribute(name, v)
      case .intArray(let v): op.updateAttribute(name, v)
      case .floatArray(let v): op.updateAttribute(name, v)
      case .doubleArray(let v): op.updateAttribute(name, v)
      case .stringArray(let v): op.updateAttribute(name, v)
      case .constTensor(_): fatalError("Const Tensor cannot be eager attribute.")
      case .tensorDataTypeValue(let v): op.updateAttribute(name, v)
      case .tensorDataTypeArray(let v): op.updateAttribute(name, v)
      case .optionalTensorShape(let v): op.updateAttribute(name, v)
      case .optionalTensorShapeArray(let v): op.updateAttribute(name, v)
      case .tensorFunctionPointer(let v): op.updateAttribute(name, v)
      }
    }
    return op
  }

  func updateOutputShapes() {
    let status = TF_NewStatus()
    defer { TF_DeleteStatus(status) }

    /// Returns shape only if it has already been computed.
    func shape(for handle: LazyTensorHandle) -> TensorShape? {
      switch handle.handle {
      case .symbolic(let op, let index, _): return op.outputShapes[index]
      case .concrete(let tfeHandle, _): return tfeHandle.shape
      }
    }

    let inputShapes: [TensorShape?] = inputs.lazy.flatMap { (input) -> [TensorShape?] in
      switch input {
      case .single(let handle): return [shape(for: handle)]
      case .list(let values): return values.lazy.map { shape(for: $0) }
      }
    }
    let inputShapeList = TF_NewShapeAndTypeList( /*num_shapes*/Int32(inputShapes.count))
    defer { TF_DeleteShapeAndTypeList(inputShapeList) }
    for (i, shape) in inputShapes.enumerated() {
      guard let shape = shape else {
        TF_ShapeAndTypeListSetUnknownShape(inputShapeList, Int32(i))
        continue
      }
      let int64_dimensions = shape.dimensions.map { Int64($0) }
      int64_dimensions.withUnsafeBufferPointer { buffer in
        TF_ShapeAndTypeListSetShape(
          inputShapeList,
          /*index*/Int32(i),
          buffer.baseAddress,
          Int32(int64_dimensions.count))
      }
    }

    // Returns the `CTensor`, selectively materializing it if needed.
    func cTensor(handle: LazyTensorHandle) -> CTensor? {
      switch handle.handle {
      case .concrete(let h, _):
        let cTensor = TFE_TensorHandleResolve(h._cTensorHandle, status)
        checkOk(status)
        return cTensor
      case .symbolic(let op, _, _):
        // TODO(https://bugs.swift.org/browse/TF-765): "Pack" is used
        // for creating tensors from array literals. So, allow
        // materialization for 'Pack' so that we can get the shape for
        // array literals. We should revisit this heuristic.
        if op.name != "Pack" { return nil }
        let cTensor = TFE_TensorHandleResolve(handle._cTensorHandle, status)
        checkOk(status)
        return cTensor
      }
    }

    // Create `inputTensors` consisting of *only* materialized inputs.
    var inputTensors: [CTensor?] = []
    for input in inputs {
      switch input {
      case .single(let v):
        inputTensors.append(cTensor(handle: v))
      case .list(let values):
        inputTensors.append(contentsOf: values.lazy.map { cTensor(handle: $0) })
      }
    }

    // This will be filled in by `TFE_InferShapes` and should be freed later.
    var outputShapeListPtr = UnsafeMutablePointer<TF_ShapeAndTypeList>(nil)
    defer { TF_DeleteShapeAndTypeList(outputShapeListPtr) }

    let tfeOp = self.tfeOp
    defer {
      TFE_DeleteOp(tfeOp.op)
      TF_DeleteStatus(tfeOp.status)
    }

    inputTensors.withUnsafeMutableBufferPointer { buffer in
      TFE_InferShapes(
        tfeOp.op,
        /*input_shapes*/inputShapeList,
        /*input_tensors*/buffer.baseAddress!,
        /*input_tensors_as_shapes*/nil,
        /*input_resource_shapes_and_types*/nil,
        /*output_shapes*/&outputShapeListPtr,
        /*output_resource_shapes_and_types*/nil,
        status)
      checkOk(status)
    }

    precondition(outputShapeListPtr != nil, "TFE_InferShapes returned nil for output shapes")
    let outputShapeList = outputShapeListPtr!.pointee
    outputShapes = (0..<outputShapeList.num_items).lazy.map { index -> TensorShape? in
      let outputShape = outputShapeList.items![Int(index)]
      if outputShape.num_dims == -1 { return nil }
      let dims = (0..<outputShape.num_dims).lazy.map { Int(outputShape.dims![Int($0)]) }
      let hasUnknownDims = dims.contains { $0 == -1 }
      return hasUnknownDims ? nil : TensorShape(dims)
    }
  }
}
