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

/// Swift-side convenience wrapper for a `TF_Graph`. It holds a pointer to the
/// underlying TF_Graph and also exposes some aspects of the TF_Graph as
/// properties.
class TFGraph {
  /// The `TF_Operation *` type.
  typealias CTFOperation = OpaquePointer

  enum Input {
    case single(TF_Output)
    case list([TF_Output])
  }

  /// The pointer to the underlying TF Graph.
  let cTFGraph: CTFGraph = TF_NewGraph()
  var inputs: [TF_Output] = []
  var nodes: [CTFOperation?] = []
  var outputs: [TF_Output] = []
  /// An array representing how the outputs are grouped. The grouping
  /// corresponds to a higher-level notion like TensorGroup.
  var outputGroupCounts: [Int] = []
  var name: String { "lazyTrace_\(nodeCount)" }

  /// A status object to pass to TF graph building operations.
  private let status: CTFStatus = TF_NewStatus()

  /// Counter that is used for number the generated graph nodes.
  private var nodeCount: Int = 0

  init(trace: LazyTensorTrace) {
    var nodesCache: [ObjectIdentifier: CTFOperation?] = [:]
    for op in trace.operations {
      let opInputs = op.inputs.map { input -> TFGraph.Input in
        switch input {
        case LazyTensorOperation.Input.single(let h):
          return TFGraph.Input.single(makeTFOutput(handle: h, nodesCache: nodesCache))
        case LazyTensorOperation.Input.list(let elements):
          let tfInputs = elements.map { makeTFOutput(handle: $0, nodesCache: nodesCache) }
          return TFGraph.Input.list(tfInputs)
        }
      }
      let graphNode = makeTFGraphNode(
        name: op.name,
        attributes: op.attributes,
        inputs: opInputs,
        device: op.deviceName)
      nodesCache[ObjectIdentifier(op)] = graphNode
      if op.name != "Placeholder" { nodes.append(graphNode) }
    }
    self.inputs = trace.inputs.map {
      TF_Output(oper: nodesCache[ObjectIdentifier($0)]!, index: 0)
    }
    for output in trace.outputs {
      let graphNode = nodesCache[ObjectIdentifier(output)]!
      outputGroupCounts.append(output.outputCount)
      outputs += (0..<output.outputCount).map {
        TF_Output(oper: graphNode, index: Int32($0))
      }
    }
  }

  deinit {
    TF_DeleteGraph(cTFGraph)
    TF_DeleteStatus(status)
  }

  private func newNodeName(base: String) -> String {
    let name = "\(base)_\(nodeCount)"
    nodeCount += 1
    return name
  }

  private func updateAttribute(
    description: CTFOperationDescription,
    name: String,
    attribute: LazyTensorOperation.Attribute
  ) {
    switch attribute {
    case .tensorDataTypeValue(let value):
      TF_SetAttrType(description, name, value._cDataType)
    case .boolValue(let value):
      TF_SetAttrBool(description, name, value ? 1 : 0)
    case .intValue(let value):
      TF_SetAttrInt(description, name, Int64(value))
    case .floatValue(let value):
      TF_SetAttrFloat(description, name, value)
    case .doubleValue(let value):
      TF_SetAttrFloat(description, name, Float(value))
    case .stringValue(let value):
      value.utf8CString.withUnsafeBufferPointer { buffer in
        // utf8CString is null-terminated; TF_SetAttrString wants
        // non-null-terminated.
        TF_SetAttrString(description, name, buffer.baseAddress, buffer.count - 1)
      }
    case .intArray(let values):
      let values64 = values.map { Int64($0) }
      values64.withUnsafeBufferPointer { buffer in
        TF_SetAttrIntList(description, name, buffer.baseAddress, Int32(buffer.count))
      }
    case .constTensor(let value):
      let cTensor = TFE_TensorHandleResolve(value._cTensorHandle, status)
      checkOk(status)
      TF_SetAttrTensor(description, name, cTensor!, status)
    case .tensorDataTypeArray(let values):
      values.withUnsafeBufferPointer { buffer in
        buffer.withMemoryRebound(to: TF_DataType.self) { reboundBuffer in
          TF_SetAttrTypeList(
            description,
            name,
            reboundBuffer.baseAddress,
            Int32(reboundBuffer.count))
        }
      }
    case .optionalTensorShape(let value):
      if let shape = value {
        let dimensions: [Int64] = shape.dimensions.map(Int64.init)
        dimensions.withUnsafeBufferPointer { buffer in
          TF_SetAttrShape(description, name, buffer.baseAddress, Int32(buffer.count))
        }
      } else {
        TF_SetAttrShape(description, name, nil, -1)
      }
    case .optionalTensorShapeArray(let values):
      let flattenedDims = values.flatMap { (tensorShapeOpt) -> [Int64] in
        if let tensorShape = tensorShapeOpt {
          return tensorShape.dimensions.map(Int64.init)
        }
        return []
      }
      let ranks = values.map { shape in (shape?.rank).map(Int32.init) ?? -1 }
      flattenedDims.withUnsafeBufferPointer { flattenedDimsBuffer in
        var dimsPtr: UnsafePointer<Int64>? = flattenedDimsBuffer.baseAddress
        var dims: [UnsafePointer<Int64>?] = []
        for rank in ranks {
          dims.append(dimsPtr)
          if rank >= 0 {
            dimsPtr = dimsPtr.map { $0.advanced(by: Int(rank)) }
          }
        }
        dims.withUnsafeMutableBufferPointer { dimsBuffer in
          ranks.withUnsafeBufferPointer { ranksBuffer in
            TF_SetAttrShapeList(
              description,
              name,
              dimsBuffer.baseAddress,
              ranksBuffer.baseAddress,
              Int32(ranksBuffer.count))
          }
        }
      }
    case .tensorFunctionPointer(let value):
      TF_SetAttrFuncName(description, name, value.name, value.name.count)
    default: fatalError("Unhandled attribute \(name):\(attribute)")
    }
  }

  private func makeTFGraphNode(
    name: String,
    attributes: [String: LazyTensorOperation.Attribute],
    inputs: [Input],
    device: String?
  ) -> CTFOperation? {
    // Create a new graph node now.
    let description: CTFOperationDescription! = TF_NewOperation(
      cTFGraph,
      name,
      newNodeName(base: name))

    // Set Attributes
    for (name, value) in attributes {
      updateAttribute(description: description, name: name, attribute: value)
    }

    // Add Inputs
    for input in inputs {
      switch input {
      case Input.single(let singleInput):
        TF_AddInput(description, singleInput)
      case Input.list(let inputList):
        inputList.withUnsafeBufferPointer { buffer in
          TF_AddInputList(description, buffer.baseAddress, Int32(buffer.count))
        }
      }
    }

    if let device = device { TF_SetDevice(description, device) }

    // Finalize operation.
    let graphNode = TF_FinishOperation(description, status)
    checkOk(status)
    return graphNode!
  }

  private func makeTFOutput(
    handle: LazyTensorHandle,
    nodesCache: [ObjectIdentifier: CTFOperation?]
  ) -> TF_Output {
    if case let .symbolic(lazyOp, index, _) = handle.handle {
      let id = ObjectIdentifier(lazyOp)
      return TF_Output(oper: nodesCache[id]!, index: Int32(index))
    }
    fatalError("Should only have symbolic inputs.")
  }
}

/// Swift-side convenience wrapper for a `TF_Function`.
class TFFunction {
  let cTFFunction: CTFFunction
  let outputCount: Int
  let outputGroupCounts: [Int]
  var name: String { String(cString: TF_FunctionName(cTFFunction)!) }

  init(trace: LazyTensorTrace, name: String? = nil) {
    let status: CTFStatus = TF_NewStatus()
    defer { TF_DeleteStatus(status) }
    let graph = TFGraph(trace: trace)
    let cTFGraph = graph.cTFGraph
    let inputs = graph.inputs
    let outputs = graph.outputs
    let tracedFnName = name ?? graph.name
    self.outputCount = outputs.count
    self.outputGroupCounts = graph.outputGroupCounts
    self.cTFFunction = graph.nodes.withUnsafeBufferPointer {
      operations -> CTFFunction in
      let base = operations.baseAddress
      let tracedGraphFn = TF_GraphToFunction(
        cTFGraph,
        tracedFnName,
        /*append_hash_to_fn_name*/(name == nil ? 1 : 0),
        /*num_opers*/Int32(operations.count),
        /*opers*/base,
        /*numinputs*/Int32(inputs.count),
        /*inputs*/inputs,
        /*noutputs*/Int32(outputs.count),
        /*outputs*/outputs,
        /*outputnames*/nil,
        /*functionoptions*/nil,
        "",
        status)
      checkOk(status)
      if _RuntimeConfig.printsDebugLog {
        var len: Int = 0
        let funcDebugStr = TF_FunctionDebugString(tracedGraphFn, &len)!
        debugLog("The traced function is:\n\(String(cString: funcDebugStr))")
        free(funcDebugStr)
        debugLog("Corresponding lazy tensor operations:\n")
        for output in graph.outputs {
          debugLog("  \(output)")
        }
      }
      return tracedGraphFn!
    }

    let eagerContext = _TFCGetGlobalEagerContext()
    TFE_ContextAddFunction(eagerContext, self.cTFFunction, status)
    checkOk(status)
  }

  func execute(_ inputs: [TFETensorHandle], usingXLA: Bool = false) -> [TFETensorHandle] {
    let status: CTFStatus = TF_NewStatus()
    defer { TF_DeleteStatus(status) }

    let eagerContext = _TFCGetGlobalEagerContext()
    let fname = TF_FunctionName(cTFFunction)!
    let eagerOp: CTFEOp! = TFE_NewOp(eagerContext, fname, status)
    defer { TFE_DeleteOp(eagerOp) }
    checkOk(status)

    let deviceName = _ExecutionContext.global.currentDeviceName
    if let deviceName = deviceName {
      debugLog("Placing the trace func on device \(deviceName).")
      TFE_OpSetDevice(eagerOp, deviceName, status)
      checkOk(status)
    }

    if usingXLA {
      debugLog("Enabling XLA compilation")
      TFE_OpSetAttrBool(eagerOp, "_XlaCompile", 1)
    }

    for input in inputs {
      TFE_OpAddInput(eagerOp, input._cTensorHandle, status)
      checkOk(status)
    }

    var returnValues = [CTensorHandle?](repeating: nil, count: outputCount)
    var outputReturnValueCount = Int32(outputCount)
    TFE_Execute(eagerOp, &returnValues, &outputReturnValueCount, status)
    checkOk(status)

    return returnValues.map { TFETensorHandle(_owning: $0!) }
  }
}

extension TFFunction: CustomStringConvertible {
  var description: String {
    var len: Int = 0
    let funcDebugStr = TF_FunctionDebugString(cTFFunction, &len)!
    let result = String(cString: funcDebugStr)
    free(funcDebugStr)
    return result
  }
}
