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

@usableFromInline
class LazyTensorHandle: _AnyTensorHandle {
  enum Handle {
    /// Bool indicates if this concrete TFETensorhandle was a result of
    /// materialization.
    case concrete(TFETensorHandle, materialized: Bool)
    /// Bool indicates whether this is a live tensor. This flag is used to
    /// heuristically determine whether this symbolic tensor should also be
    /// materialized whenever materialization of any other tensor is triggered.
    case symbolic(LazyTensorOperation, index: Int, isLive: Bool)
  }

  let handle: Handle

  @usableFromInline
  var _tfeTensorHandle: TFETensorHandle {
    switch handle {
    case .concrete(let h, _):
      return h
    case .symbolic(let op, let index, _):
      return op.materialized(index: index)
    }
  }

  init(_ base: TFETensorHandle) {
    handle = Handle.concrete(base, materialized: false)
  }

  init(_materialized base: TFETensorHandle) {
    handle = Handle.concrete(base, materialized: true)
  }

  init(_lazy op: LazyTensorOperation, index: Int) {
    precondition(
      index < op.outputCount, "Symbolic Tensor Index is out-of-bounds")
    handle = Handle.symbolic(op, index: index, isLive: false)
    LazyTensorContext.local.operationsTracker.incrementRefCount(op, isLive: false)
  }

  init(_lazyLive op: LazyTensorOperation, index: Int) {
    precondition(
      index < op.outputCount, "Symbolic Tensor Index is out-of-bounds")
    handle = Handle.symbolic(op, index: index, isLive: true)
    LazyTensorContext.local.operationsTracker.incrementRefCount(op, isLive: true)
  }

  deinit {
    if case let .symbolic(op, _, isLive) = handle {
      LazyTensorContext.local.operationsTracker.decrementRefCount(op, isLive: isLive)
    }
  }

  /// The number of dimensions of the underlying `Tensor`.
  @usableFromInline
  var rank: Int {
    @_semantics("autodiff.nonvarying")
    get { shape.rank }
  }

  /// The shape of the underlying `Tensor`.
  @usableFromInline
  var shape: TensorShape {
    @_semantics("autodiff.nonvarying")
    get {
      switch handle {
      case .symbolic(let op, let index, _):
        precondition(
          LazyTensorContext.local.isShapeTrackingEnabled,
          "Shape tracking is not enabled in this context.")
        if let shape = op.outputShapes[index] { return shape }
        // Materialize and get the shape from concrete tensor handle.
        op.outputShapes[index] = _tfeTensorHandle.shape
        return op.outputShapes[index]!
      case .concrete(let tfeHandle, _): return tfeHandle.shape
      }
    }
  }

  /// Returns the underlying `LazyTensorOperation` if this is a symbolic `LazyTensorHandle`.
  var lazyTensorOperation: LazyTensorOperation? {
    switch handle {
    case .symbolic(let op, _, _): return op
    case .concrete: return nil
    }
  }

  @usableFromInline
  var backend: Device.Backend { return .TF_EAGER }

  // Liveness tracking for LazyTensorOperations
  //
  static func isLive(_ op: LazyTensorOperation) -> Bool {
    return LazyTensorContext.local.operationsTracker.isLive(op)
  }

  static func forEachLiveOperation(
    _ perform: (LazyTensorOperation) throws -> Void
  ) rethrows {
    try LazyTensorContext.local.operationsTracker.forEachLiveOperation(perform)
  }

  static func forEachOperation(
    _ perform: (LazyTensorOperation) throws -> Void
  ) rethrows {
    try LazyTensorContext.local.operationsTracker.forEachOperation(perform)
  }

  @usableFromInline
  static var _materializationCallback: (String) -> Void = { _ in }
}

extension _AnyTensorHandle {
  /// Returns a concrete `LazyTensorHandle` with an additional constraint that the
  /// underlying concrete `LazyTensorHandle` should be marked to be promoted as an
  /// input when used in an extracted trace.  This provides a **temporary**
  /// mechanism to promote a concrete lazy tensor to an input in extracted
  /// traces. (Note that this may trigger materialization.)
  var _concreteInputLazyTensor: LazyTensorHandle {
    LazyTensorHandle(_materialized: self._tfeTensorHandle)
  }
}

extension TensorHandle {
  /// Returns `Self` that wraps `_concreteInputLazyTensor` of the underlying
  /// `_AnyTensorHandle`
  public var _concreteInputLazyTensor: TensorHandle {
    TensorHandle(handle: handle._concreteInputLazyTensor)
  }
}

extension Tensor {
  /// Returns `Self` that wraps `_concreteInputLazyTensor` of the underlying
  /// `_AnyTensorHandle`
  public var _concreteInputLazyTensor: Tensor {
    Tensor(handle: handle._concreteInputLazyTensor)
  }
}

extension StringTensor {
  /// Returns `Self` that wraps `_concreteInputLazyTensor` of the underlying
  /// `_AnyTensorHandle`
  public var _concreteInputLazyTensor: StringTensor {
    StringTensor(handle: handle._concreteInputLazyTensor)
  }
}

extension VariantHandle {
  /// Returns `Self` that wraps `_concreteInputLazyTensor` of the underlying
  /// `_AnyTensorHandle`
  public var _concreteInputLazyTensor: VariantHandle {
    VariantHandle(handle: handle._concreteInputLazyTensor)
  }
}

extension ResourceHandle {
  /// Returns `Self` that wraps `_concreteInputLazyTensor` of the underlying
  /// `_AnyTensorHandle`
  public var _concreteInputLazyTensor: ResourceHandle {
    ResourceHandle(handle: handle._concreteInputLazyTensor)
  }
}

class LazyTensorOperation: TensorOperation {
  typealias TensorValueHandle = LazyTensorHandle

  enum Input {
    case single(LazyTensorHandle)
    case list([LazyTensorHandle])
  }

  enum Attribute: Equatable {
    case boolValue(Bool)
    case intValue(Int)
    case floatValue(Float)
    case doubleValue(Double)
    case stringValue(String)
    case boolArray([Bool])
    case intArray([Int])
    case floatArray([Float])
    case doubleArray([Double])
    case stringArray([String])
    case constTensor(TFETensorHandle)
    case tensorDataTypeValue(TensorDataType)
    case tensorFunctionPointer(_TensorFunctionPointer)
    case tensorDataTypeArray([TensorDataType])
    case optionalTensorShape(TensorShape?)
    case optionalTensorShapeArray([TensorShape?])
  }

  var name: String
  let outputCount: Int
  var inputs: [Input]
  var attributes: [String: Attribute]
  var outputShapes: [TensorShape?]
  var deviceName: String?
  var outputs: [TFETensorHandle]?
  var id: String?

  var nameWithID: String {
    if let id = self.id {
      return "\(name)_\(id)"
    } else {
      return "\(name)_\(ObjectIdentifier(self))"
    }
  }

  func outputName(at index: Int) -> String {
    precondition(
      index < outputCount,
      "Output index out of bounds when getting outputName.")
    let ssaID = id ?? "\(ObjectIdentifier(self))"
    var ssaName = "%\(ssaID)"
    if outputCount > 1 {
      ssaName += ".\(index)"
    }
    return ssaName
  }

  var outputName: String {
    switch outputCount {
    case 0: return ""
    case 1: return outputName(at: 0)
    default:
      let outputNames = (0..<outputCount).lazy.map {
        self.outputName(at: $0)
      }
      let aggregateName = outputNames.joined(separator: ", ")
      return "(\(aggregateName))"
    }
  }

  static var liveOperations: Int = 0

  init(_id id: String?, name: String, outputCount: Int) {
    self.name = name
    self.inputs = []
    self.attributes = [:]
    self.deviceName = _ExecutionContext.global.currentDeviceName
    self.outputCount = outputCount
    self.outputShapes = []
    self.outputs = nil
    self.id = id
    LazyTensorOperation.liveOperations += 1
  }

  required convenience init(_ name: String, _ outputCount: Int) {
    self.init(_id: nil, name: name, outputCount: outputCount)
  }

  deinit {
    LazyTensorOperation.liveOperations -= 1
  }

  func evaluate() -> [LazyTensorHandle] {
    if LazyTensorContext.local.isShapeTrackingEnabled {
      updateOutputShapes()
    }
    return (0..<outputCount).map {
      LazyTensorHandle(_lazyLive: self, index: $0)
    }
  }

  func addInput(_ input: LazyTensorHandle) {
    inputs.append(Input.single(input))
  }

  func updateAttribute(_ name: String, _ value: Bool) {
    attributes[name] = Attribute.boolValue(value)
  }
  func updateAttribute(_ name: String, _ value: Int) {
    attributes[name] = Attribute.intValue(value)
  }
  func updateAttribute(_ name: String, _ value: Int32) {
    attributes[name] = Attribute.intValue(Int(value))
  }
  func updateAttribute(_ name: String, _ value: Int64) {
    attributes[name] = Attribute.intValue(Int(value))
  }
  func updateAttribute(_ name: String, _ value: Float) {
    attributes[name] = Attribute.floatValue(value)
  }
  func updateAttribute(_ name: String, _ value: Double) {
    attributes[name] = Attribute.doubleValue(value)
  }
  func updateAttribute(_ name: String, _ value: String) {
    attributes[name] = Attribute.stringValue(value)
  }
  func updateAttribute(_ name: String, _ value: [Bool]) {
    attributes[name] = Attribute.boolArray(value)
  }
  func updateAttribute(_ name: String, _ value: [Int]) {
    attributes[name] = Attribute.intArray(value)
  }
  func updateAttribute(_ name: String, _ value: [Int32]) {
    attributes[name] = Attribute.intArray(value.map { Int($0) })
  }
  func updateAttribute(_ name: String, _ value: [Int64]) {
    attributes[name] = Attribute.intArray(value.map { Int($0) })
  }
  func updateAttribute(_ name: String, _ value: [Float]) {
    attributes[name] = Attribute.floatArray(value)
  }
  func updateAttribute(_ name: String, _ value: [Double]) {
    attributes[name] = Attribute.doubleArray(value)
  }
  func updateAttribute(_ name: String, _ value: [String]) {
    attributes[name] = Attribute.stringArray(value)
  }
}

extension LazyTensorOperation: TFTensorOperation {
  private func lazyTensorHandle(_ input: _AnyTensorHandle) -> LazyTensorHandle {
    if let lazyHandle = input as? LazyTensorHandle {
      if case let LazyTensorHandle.Handle.symbolic(
        op, index, true) = lazyHandle.handle
      {
        // We turn off liveness for the constructed LazyTensorHandle,
        // because it is only referenced internally as a part
        // of the LazyTensorOperation input.
        return LazyTensorHandle(_lazy: op, index: index)
      } else {
        return lazyHandle
      }
    } else {
      return LazyTensorHandle(input._tfeTensorHandle)
    }
  }

  func addInput(_ input: _AnyTensorHandle) {
    addInput(lazyTensorHandle(input))
  }

  func addInput<Scalar: TensorFlowScalar>(_ input: Tensor<Scalar>) {
    addInput(input.handle.handle)
  }

  func addInput(_ input: StringTensor) {
    addInput(input.handle.handle)
  }

  func addInput(_ input: VariantHandle) {
    addInput(input.handle)
  }

  func addInput(_ input: ResourceHandle) {
    addInput(input.handle)
  }

  func addInputList<T: TensorArrayProtocol>(_ input: T) {
    let lazyHandles = input._tensorHandles.map { lazyTensorHandle($0) }
    inputs.append(Input.list(lazyHandles))
  }

  func updateAttribute(_ name: String, _ value: TensorDataType) {
    attributes[name] = Attribute.tensorDataTypeValue(value)
  }
  func updateAttribute(_ name: String, _ value: TensorShape) {
    attributes[name] = Attribute.optionalTensorShape(value)
  }
  func updateAttribute(_ name: String, _ value: TensorShape?) {
    attributes[name] = Attribute.optionalTensorShape(value)
  }
  func updateAttribute(_ name: String, _ value: [TensorDataType]) {
    attributes[name] = Attribute.tensorDataTypeArray(value)
  }
  func updateAttribute(_ name: String, _ value: [TensorShape]) {
    attributes[name] = Attribute.optionalTensorShapeArray(value)
  }
  func updateAttribute(_ name: String, _ value: [TensorShape?]) {
    attributes[name] = Attribute.optionalTensorShapeArray(value)
  }
  func updateAttribute(_ name: String, _ value: _TensorFunctionPointer) {
    attributes[name] = Attribute.tensorFunctionPointer(value)
  }
  func updateAttribute(_ name: String, _ value: TFETensorHandle) {
    attributes[name] = Attribute.constTensor(value)
  }

  func updateAttribute<In: TensorGroup, Out: TensorGroup>(
    _ name: String, _ value: (In) -> Out
  ) {
    updateAttribute(name, _TensorFunctionPointer(name: _tffunc(value)))
  }

  func execute() {
    // If we want to stage this, we will need to add control dependencies.
    // For the time-being, just build a TFE_Op and run it.
    //
    // Collect all the unmaterialized inputs.
    var unmaterializedInputs = [LazyTensorOperation]()
    unmaterializedInputs.reserveCapacity(inputs.count)
    for input in inputs {
      switch input {
      case .single(let v):
        if let lazyOperation = v.lazyTensorOperation {
          unmaterializedInputs.append(lazyOperation)
        }
      case .list(let values):
        unmaterializedInputs.append(
          contentsOf: values.lazy.compactMap { $0.lazyTensorOperation }
        )
      }
    }
    // Materialize the inputs now.
    LazyTensorOperation.materialize(targets: unmaterializedInputs)

    // Build the TFEOp and execute.
    let op = TFE_Op(name, outputCount)
    for input in inputs {
      switch input {
      case .single(let v):
        op.addInput(v._tfeTensorHandle)
      case .list(let values):
        for v in values {
          op.addInput(v._tfeTensorHandle)
        }
      }
    }
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
    op.execute()
  }

  func execute<T0: TensorArrayProtocol>(
    _ count0: Int
  ) -> (T0) {
    let outputs = evaluate()
    let offset0 = 0
    let result = (T0.init(_handles: outputs[offset0..<count0]))
    return result
  }

  func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol>(
    _ count0: Int,
    _ count1: Int
  ) -> (T0, T1) {
    let outputs = evaluate()
    let offset0 = 0
    let offset1 = offset0 + count0
    let result = (
      T0.init(_handles: outputs[offset0..<offset1]),
      T1.init(_handles: outputs[offset1..<outputs.count])
    )
    return result
  }

  func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol>(
    _ count0: Int,
    _ count1: Int,
    _ count2: Int
  ) -> (T0, T1, T2) {
    let outputs = evaluate()
    let offset0 = 0
    let offset1 = offset0 + count0
    let offset2 = offset1 + count1
    let result = (
      T0.init(_handles: outputs[offset0..<offset1]),
      T1.init(_handles: outputs[offset1..<offset2]),
      T2.init(_handles: outputs[offset2..<outputs.count])
    )
    return result
  }

  func execute<
    T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol,
    T3: TensorArrayProtocol
  >(
    _ count0: Int,
    _ count1: Int,
    _ count2: Int,
    _ count3: Int
  ) -> (T0, T1, T2, T3) {
    let outputs = evaluate()
    let offset0 = 0
    let offset1 = offset0 + count0
    let offset2 = offset1 + count1
    let offset3 = offset2 + count2
    let result = (
      T0.init(_handles: outputs[offset0..<offset1]),
      T1.init(_handles: outputs[offset1..<offset2]),
      T2.init(_handles: outputs[offset2..<offset3]),
      T3.init(_handles: outputs[offset3..<outputs.count])
    )
    return result
  }

  func execute<
    T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol,
    T3: TensorArrayProtocol, T4: TensorArrayProtocol
  >(
    _ count0: Int,
    _ count1: Int,
    _ count2: Int,
    _ count3: Int,
    _ count4: Int
  ) -> (T0, T1, T2, T3, T4) {
    let outputs = evaluate()
    let offset0 = 0
    let offset1 = offset0 + count0
    let offset2 = offset1 + count1
    let offset3 = offset2 + count2
    let offset4 = offset3 + count3
    let result = (
      T0.init(_handles: outputs[offset0..<offset1]),
      T1.init(_handles: outputs[offset1..<offset2]),
      T2.init(_handles: outputs[offset2..<offset3]),
      T3.init(_handles: outputs[offset3..<offset4]),
      T4.init(_handles: outputs[offset4..<outputs.count])
    )
    return result
  }

  func execute<
    T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol,
    T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol
  >(
    _ count0: Int,
    _ count1: Int,
    _ count2: Int,
    _ count3: Int,
    _ count4: Int,
    _ count5: Int
  ) -> (T0, T1, T2, T3, T4, T5) {
    let outputs = evaluate()
    let offset0 = 0
    let offset1 = offset0 + count0
    let offset2 = offset1 + count1
    let offset3 = offset2 + count2
    let offset4 = offset3 + count3
    let offset5 = offset4 + count4
    let result = (
      T0.init(_handles: outputs[offset0..<offset1]),
      T1.init(_handles: outputs[offset1..<offset2]),
      T2.init(_handles: outputs[offset2..<offset3]),
      T3.init(_handles: outputs[offset3..<offset4]),
      T4.init(_handles: outputs[offset4..<offset5]),
      T5.init(_handles: outputs[offset5..<outputs.count])
    )
    return result
  }

  func execute<
    T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol,
    T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol,
    T6: TensorArrayProtocol
  >(
    _ count0: Int,
    _ count1: Int,
    _ count2: Int,
    _ count3: Int,
    _ count4: Int,
    _ count5: Int,
    _ count6: Int
  ) -> (T0, T1, T2, T3, T4, T5, T6) {
    let outputs = evaluate()
    let offset0 = 0
    let offset1 = offset0 + count0
    let offset2 = offset1 + count1
    let offset3 = offset2 + count2
    let offset4 = offset3 + count3
    let offset5 = offset4 + count4
    let offset6 = offset5 + count5
    let result = (
      T0.init(_handles: outputs[offset0..<offset1]),
      T1.init(_handles: outputs[offset1..<offset2]),
      T2.init(_handles: outputs[offset2..<offset3]),
      T3.init(_handles: outputs[offset3..<offset4]),
      T4.init(_handles: outputs[offset4..<offset5]),
      T5.init(_handles: outputs[offset5..<offset6]),
      T6.init(_handles: outputs[offset6..<outputs.count])
    )
    return result
  }

  func execute<
    T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol,
    T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol,
    T6: TensorArrayProtocol, T7: TensorArrayProtocol
  >(
    _ count0: Int,
    _ count1: Int,
    _ count2: Int,
    _ count3: Int,
    _ count4: Int,
    _ count5: Int,
    _ count6: Int,
    _ count7: Int
  ) -> (T0, T1, T2, T3, T4, T5, T6, T7) {
    let outputs = evaluate()
    let offset0 = 0
    let offset1 = offset0 + count0
    let offset2 = offset1 + count1
    let offset3 = offset2 + count2
    let offset4 = offset3 + count3
    let offset5 = offset4 + count4
    let offset6 = offset5 + count5
    let offset7 = offset6 + count6
    let result = (
      T0.init(_handles: outputs[offset0..<offset1]),
      T1.init(_handles: outputs[offset1..<offset2]),
      T2.init(_handles: outputs[offset2..<offset3]),
      T3.init(_handles: outputs[offset3..<offset4]),
      T4.init(_handles: outputs[offset4..<offset5]),
      T5.init(_handles: outputs[offset5..<offset6]),
      T6.init(_handles: outputs[offset6..<offset7]),
      T7.init(_handles: outputs[offset7..<outputs.count])
    )
    return result
  }

  func execute<
    T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol,
    T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol,
    T6: TensorArrayProtocol, T7: TensorArrayProtocol, T8: TensorArrayProtocol
  >(
    _ count0: Int,
    _ count1: Int,
    _ count2: Int,
    _ count3: Int,
    _ count4: Int,
    _ count5: Int,
    _ count6: Int,
    _ count7: Int,
    _ count8: Int
  ) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8) {
    let outputs = evaluate()
    let offset0 = 0
    let offset1 = offset0 + count0
    let offset2 = offset1 + count1
    let offset3 = offset2 + count2
    let offset4 = offset3 + count3
    let offset5 = offset4 + count4
    let offset6 = offset5 + count5
    let offset7 = offset6 + count6
    let offset8 = offset7 + count7
    let result = (
      T0.init(_handles: outputs[offset0..<offset1]),
      T1.init(_handles: outputs[offset1..<offset2]),
      T2.init(_handles: outputs[offset2..<offset3]),
      T3.init(_handles: outputs[offset3..<offset4]),
      T4.init(_handles: outputs[offset4..<offset5]),
      T5.init(_handles: outputs[offset5..<offset6]),
      T6.init(_handles: outputs[offset6..<offset7]),
      T7.init(_handles: outputs[offset7..<offset8]),
      T8.init(_handles: outputs[offset8..<outputs.count])
    )
    return result
  }

  func execute<
    T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol,
    T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol,
    T6: TensorArrayProtocol, T7: TensorArrayProtocol, T8: TensorArrayProtocol,
    T9: TensorArrayProtocol
  >(
    _ count0: Int,
    _ count1: Int,
    _ count2: Int,
    _ count3: Int,
    _ count4: Int,
    _ count5: Int,
    _ count6: Int,
    _ count7: Int,
    _ count8: Int,
    _ count9: Int
  ) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9) {
    let outputs = evaluate()
    let offset0 = 0
    let offset1 = offset0 + count0
    let offset2 = offset1 + count1
    let offset3 = offset2 + count2
    let offset4 = offset3 + count3
    let offset5 = offset4 + count4
    let offset6 = offset5 + count5
    let offset7 = offset6 + count6
    let offset8 = offset7 + count7
    let offset9 = offset8 + count8
    let result = (
      T0.init(_handles: outputs[offset0..<offset1]),
      T1.init(_handles: outputs[offset1..<offset2]),
      T2.init(_handles: outputs[offset2..<offset3]),
      T3.init(_handles: outputs[offset3..<offset4]),
      T4.init(_handles: outputs[offset4..<offset5]),
      T5.init(_handles: outputs[offset5..<offset6]),
      T6.init(_handles: outputs[offset6..<offset7]),
      T7.init(_handles: outputs[offset7..<offset8]),
      T8.init(_handles: outputs[offset8..<offset9]),
      T9.init(_handles: outputs[offset9..<outputs.count])
    )
    return result
  }

  func execute<
    T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol,
    T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol,
    T6: TensorArrayProtocol, T7: TensorArrayProtocol, T8: TensorArrayProtocol,
    T9: TensorArrayProtocol, T10: TensorArrayProtocol
  >(
    _ count0: Int,
    _ count1: Int,
    _ count2: Int,
    _ count3: Int,
    _ count4: Int,
    _ count5: Int,
    _ count6: Int,
    _ count7: Int,
    _ count8: Int,
    _ count9: Int,
    _ count10: Int
  ) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10) {
    let outputs = evaluate()
    let offset0 = 0
    let offset1 = offset0 + count0
    let offset2 = offset1 + count1
    let offset3 = offset2 + count2
    let offset4 = offset3 + count3
    let offset5 = offset4 + count4
    let offset6 = offset5 + count5
    let offset7 = offset6 + count6
    let offset8 = offset7 + count7
    let offset9 = offset8 + count8
    let offset10 = offset9 + count9
    let result = (
      T0.init(_handles: outputs[offset0..<offset1]),
      T1.init(_handles: outputs[offset1..<offset2]),
      T2.init(_handles: outputs[offset2..<offset3]),
      T3.init(_handles: outputs[offset3..<offset4]),
      T4.init(_handles: outputs[offset4..<offset5]),
      T5.init(_handles: outputs[offset5..<offset6]),
      T6.init(_handles: outputs[offset6..<offset7]),
      T7.init(_handles: outputs[offset7..<offset8]),
      T8.init(_handles: outputs[offset8..<offset9]),
      T9.init(_handles: outputs[offset9..<offset10]),
      T10.init(_handles: outputs[offset10..<outputs.count])
    )
    return result
  }

  func execute<
    T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol,
    T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol,
    T6: TensorArrayProtocol, T7: TensorArrayProtocol, T8: TensorArrayProtocol,
    T9: TensorArrayProtocol, T10: TensorArrayProtocol, T11: TensorArrayProtocol
  >(
    _ count0: Int,
    _ count1: Int,
    _ count2: Int,
    _ count3: Int,
    _ count4: Int,
    _ count5: Int,
    _ count6: Int,
    _ count7: Int,
    _ count8: Int,
    _ count9: Int,
    _ count10: Int,
    _ count11: Int
  ) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11) {
    let outputs = evaluate()
    let offset0 = 0
    let offset1 = offset0 + count0
    let offset2 = offset1 + count1
    let offset3 = offset2 + count2
    let offset4 = offset3 + count3
    let offset5 = offset4 + count4
    let offset6 = offset5 + count5
    let offset7 = offset6 + count6
    let offset8 = offset7 + count7
    let offset9 = offset8 + count8
    let offset10 = offset9 + count9
    let offset11 = offset10 + count10
    let result = (
      T0.init(_handles: outputs[offset0..<offset1]),
      T1.init(_handles: outputs[offset1..<offset2]),
      T2.init(_handles: outputs[offset2..<offset3]),
      T3.init(_handles: outputs[offset3..<offset4]),
      T4.init(_handles: outputs[offset4..<offset5]),
      T5.init(_handles: outputs[offset5..<offset6]),
      T6.init(_handles: outputs[offset6..<offset7]),
      T7.init(_handles: outputs[offset7..<offset8]),
      T8.init(_handles: outputs[offset8..<offset9]),
      T9.init(_handles: outputs[offset9..<offset10]),
      T10.init(_handles: outputs[offset10..<offset11]),
      T11.init(_handles: outputs[offset11..<outputs.count])
    )
    return result
  }

  func execute<
    T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol,
    T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol,
    T6: TensorArrayProtocol, T7: TensorArrayProtocol, T8: TensorArrayProtocol,
    T9: TensorArrayProtocol, T10: TensorArrayProtocol, T11: TensorArrayProtocol,
    T12: TensorArrayProtocol
  >(
    _ count0: Int,
    _ count1: Int,
    _ count2: Int,
    _ count3: Int,
    _ count4: Int,
    _ count5: Int,
    _ count6: Int,
    _ count7: Int,
    _ count8: Int,
    _ count9: Int,
    _ count10: Int,
    _ count11: Int,
    _ count12: Int
  ) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12) {
    let outputs = evaluate()
    let offset0 = 0
    let offset1 = offset0 + count0
    let offset2 = offset1 + count1
    let offset3 = offset2 + count2
    let offset4 = offset3 + count3
    let offset5 = offset4 + count4
    let offset6 = offset5 + count5
    let offset7 = offset6 + count6
    let offset8 = offset7 + count7
    let offset9 = offset8 + count8
    let offset10 = offset9 + count9
    let offset11 = offset10 + count10
    let offset12 = offset11 + count11
    let result = (
      T0.init(_handles: outputs[offset0..<offset1]),
      T1.init(_handles: outputs[offset1..<offset2]),
      T2.init(_handles: outputs[offset2..<offset3]),
      T3.init(_handles: outputs[offset3..<offset4]),
      T4.init(_handles: outputs[offset4..<offset5]),
      T5.init(_handles: outputs[offset5..<offset6]),
      T6.init(_handles: outputs[offset6..<offset7]),
      T7.init(_handles: outputs[offset7..<offset8]),
      T8.init(_handles: outputs[offset8..<offset9]),
      T9.init(_handles: outputs[offset9..<offset10]),
      T10.init(_handles: outputs[offset10..<offset11]),
      T11.init(_handles: outputs[offset11..<offset12]),
      T12.init(_handles: outputs[offset12..<outputs.count])
    )
    return result
  }

  func execute<
    T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol,
    T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol,
    T6: TensorArrayProtocol, T7: TensorArrayProtocol, T8: TensorArrayProtocol,
    T9: TensorArrayProtocol, T10: TensorArrayProtocol, T11: TensorArrayProtocol,
    T12: TensorArrayProtocol, T13: TensorArrayProtocol
  >(
    _ count0: Int,
    _ count1: Int,
    _ count2: Int,
    _ count3: Int,
    _ count4: Int,
    _ count5: Int,
    _ count6: Int,
    _ count7: Int,
    _ count8: Int,
    _ count9: Int,
    _ count10: Int,
    _ count11: Int,
    _ count12: Int,
    _ count13: Int
  ) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13) {
    let outputs = evaluate()
    let offset0 = 0
    let offset1 = offset0 + count0
    let offset2 = offset1 + count1
    let offset3 = offset2 + count2
    let offset4 = offset3 + count3
    let offset5 = offset4 + count4
    let offset6 = offset5 + count5
    let offset7 = offset6 + count6
    let offset8 = offset7 + count7
    let offset9 = offset8 + count8
    let offset10 = offset9 + count9
    let offset11 = offset10 + count10
    let offset12 = offset11 + count11
    let offset13 = offset12 + count12
    let result = (
      T0.init(_handles: outputs[offset0..<offset1]),
      T1.init(_handles: outputs[offset1..<offset2]),
      T2.init(_handles: outputs[offset2..<offset3]),
      T3.init(_handles: outputs[offset3..<offset4]),
      T4.init(_handles: outputs[offset4..<offset5]),
      T5.init(_handles: outputs[offset5..<offset6]),
      T6.init(_handles: outputs[offset6..<offset7]),
      T7.init(_handles: outputs[offset7..<offset8]),
      T8.init(_handles: outputs[offset8..<offset9]),
      T9.init(_handles: outputs[offset9..<offset10]),
      T10.init(_handles: outputs[offset10..<offset11]),
      T11.init(_handles: outputs[offset11..<offset12]),
      T12.init(_handles: outputs[offset12..<offset13]),
      T13.init(_handles: outputs[offset13..<outputs.count])
    )
    return result
  }

  func execute<
    T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol,
    T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol,
    T6: TensorArrayProtocol, T7: TensorArrayProtocol, T8: TensorArrayProtocol,
    T9: TensorArrayProtocol, T10: TensorArrayProtocol, T11: TensorArrayProtocol,
    T12: TensorArrayProtocol, T13: TensorArrayProtocol, T14: TensorArrayProtocol
  >(
    _ count0: Int,
    _ count1: Int,
    _ count2: Int,
    _ count3: Int,
    _ count4: Int,
    _ count5: Int,
    _ count6: Int,
    _ count7: Int,
    _ count8: Int,
    _ count9: Int,
    _ count10: Int,
    _ count11: Int,
    _ count12: Int,
    _ count13: Int,
    _ count14: Int
  ) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14) {
    let outputs = evaluate()
    let offset0 = 0
    let offset1 = offset0 + count0
    let offset2 = offset1 + count1
    let offset3 = offset2 + count2
    let offset4 = offset3 + count3
    let offset5 = offset4 + count4
    let offset6 = offset5 + count5
    let offset7 = offset6 + count6
    let offset8 = offset7 + count7
    let offset9 = offset8 + count8
    let offset10 = offset9 + count9
    let offset11 = offset10 + count10
    let offset12 = offset11 + count11
    let offset13 = offset12 + count12
    let offset14 = offset13 + count13
    let result = (
      T0.init(_handles: outputs[offset0..<offset1]),
      T1.init(_handles: outputs[offset1..<offset2]),
      T2.init(_handles: outputs[offset2..<offset3]),
      T3.init(_handles: outputs[offset3..<offset4]),
      T4.init(_handles: outputs[offset4..<offset5]),
      T5.init(_handles: outputs[offset5..<offset6]),
      T6.init(_handles: outputs[offset6..<offset7]),
      T7.init(_handles: outputs[offset7..<offset8]),
      T8.init(_handles: outputs[offset8..<offset9]),
      T9.init(_handles: outputs[offset9..<offset10]),
      T10.init(_handles: outputs[offset10..<offset11]),
      T11.init(_handles: outputs[offset11..<offset12]),
      T12.init(_handles: outputs[offset12..<offset13]),
      T13.init(_handles: outputs[offset13..<offset14]),
      T14.init(_handles: outputs[offset14..<outputs.count])
    )
    return result
  }
}

extension TFETensorHandle {
  public var valueDescription: String {
    let dtype = TFE_TensorHandleDataType(self._cTensorHandle)
    switch dtype {
    case TF_FLOAT:
      return Tensor(handle: TensorHandle<Float>(handle: self)).description
    case TF_DOUBLE:
      return Tensor(handle: TensorHandle<Double>(handle: self)).description
    case TF_BFLOAT16:
      return Tensor(handle: TensorHandle<BFloat16>(handle: self)).description
    case TF_INT64:
      return Tensor(handle: TensorHandle<Int64>(handle: self)).description
    case TF_INT32:
      return Tensor(handle: TensorHandle<Int32>(handle: self)).description
    case TF_INT16:
      return Tensor(handle: TensorHandle<Int16>(handle: self)).description
    case TF_INT8:
      return Tensor(handle: TensorHandle<Int8>(handle: self)).description
    case TF_UINT64:
      return Tensor(handle: TensorHandle<UInt64>(handle: self)).description
    case TF_UINT32:
      return Tensor(handle: TensorHandle<UInt32>(handle: self)).description
    case TF_UINT16:
      return Tensor(handle: TensorHandle<UInt16>(handle: self)).description
    case TF_UINT8:
      return Tensor(handle: TensorHandle<UInt8>(handle: self)).description
    case TF_BOOL:
      return Tensor(handle: TensorHandle<Bool>(handle: self)).description
    case TF_STRING:
      // TODO(https://bugs.swift.org/browse/TF-561): The current
      // implementation of ShapedArray<String> is not correct, which
      // causes seg faults.
      return "\"string\""
    default:
      return TFETensorHandle.tfDataTypeAsString(dtype)
    }
  }

  static func tfDataTypeAsString(_ cDataType: TF_DataType) -> String {
    switch cDataType {
    case TF_FLOAT: return "float"
    case TF_DOUBLE: return "double"
    case TF_INT32: return "int32"
    case TF_UINT8: return "uint8"
    case TF_INT16: return "int16"
    case TF_INT8: return "int8"
    case TF_STRING: return "string"
    case TF_COMPLEX64, TF_COMPLEX: return "complex"
    case TF_INT64: return "int64"
    case TF_BOOL: return "bool"
    case TF_QINT8: return "qint8"
    case TF_QUINT8: return "quint8"
    case TF_QINT32: return "qint32"
    case TF_BFLOAT16: return "bfloat16"
    case TF_QINT16: return "qint16"
    case TF_QUINT16: return "quint16"
    case TF_UINT16: return "uint16"
    case TF_COMPLEX128: return "complex128"
    case TF_HALF: return "half"
    case TF_RESOURCE: return "resource"
    case TF_VARIANT: return "variant"
    case TF_UINT32: return "uint32"
    case TF_UINT64: return "uint64"
    default: fatalError("Unhandled type: \(cDataType)")
    }
  }
}

extension LazyTensorOperation.Attribute: CustomStringConvertible {
  var description: String {
    switch self {
    case .boolValue(let v): return "\(v)"
    case .intValue(let v): return "Int(\(v))"
    case .floatValue(let v): return "Float(\(v))"
    case .doubleValue(let v): return "Double(\(v))"
    case .stringValue(let v): return "\"\(v)\""
    case .boolArray(let values): return arrayAsString("", values)
    case .intArray(let values): return arrayAsString("Int", values)
    case .floatArray(let values): return arrayAsString("Float", values)
    case .doubleArray(let values): return arrayAsString("Double", values)
    case .stringArray(let values): return arrayAsString("String", values)
    case .constTensor(let v): return v.valueDescription
    case .tensorDataTypeValue(let v): return dataTypeAsString(v)
    case .tensorFunctionPointer(let v): return "TFFunction(\(v.name))"
    case .tensorDataTypeArray(let values):
      let descriptions = values.map { dataTypeAsString($0) }
      let descString = descriptions.joined(separator: ", ")
      return "[\(descString)]"
    case .optionalTensorShape(let t): return String(describing: t)
    case .optionalTensorShapeArray(let t): return "\(t)"
    }
  }

  private func arrayAsString<T>(_ desc: String, _ values: [T]) -> String {
    let arrayDesc = (values.map { "\($0)" }).joined(separator: ", ")
    return "\(desc)[\(arrayDesc)]"
  }

  private func dataTypeAsString(_ dataType: TensorDataType) -> String {
    return TFETensorHandle.tfDataTypeAsString(dataType._cDataType)
  }
}

extension LazyTensorHandle: CustomStringConvertible {
  public var description: String {
    switch self.handle {
    case .concrete(let h, let isMaterialized):
      return isMaterialized
        ? "\(h.valueDescription)*"
        : "\(h.valueDescription)"
    case .symbolic(let op, let index, let isLive):
      return op.outputName(at: index) + (isLive ? "*" : "")
    }
  }
}

extension LazyTensorOperation: CustomStringConvertible {
  public var description: String {
    let attributesDesc = attributes.sorted(by: { $0.key < $1.key }).map { "\($0): \($1)" }
    let inputsDesc = inputs.map { input -> String in
      switch input {
      case Input.single(let lazyTensor):
        return "\(lazyTensor)"
      case Input.list(let lazyTensorList):
        let lazyTensors = lazyTensorList.map { "\($0)" }
        let lazyTensorsDesc = lazyTensors.joined(separator: ", ")
        return "[\(lazyTensorsDesc)]"
      }
    }
    var desc = "\(outputName) = \(name)"
    if !attributes.isEmpty {
      desc += "["
      desc += attributesDesc.joined(separator: ", ")
      desc += "]"
    }
    desc += "("
    desc += inputsDesc.joined(separator: ", ")
    desc += ")"
    return desc
  }
}

extension LazyTensorOperation {
  /// Returns the materialized value at the given output `index`.
  func materialized(index: Int) -> TFETensorHandle {
    precondition(index < outputCount)
    return materialized()[index]
  }

  /// Materializes all the outputs.
  func materialized() -> [TFETensorHandle] {
    // Return materialized outputs if any.
    if let outputs = outputs { return outputs }

    LazyTensorOperation.materialize(targets: [self])

    // Our outputs should have been updated by now. Otherwise,
    // something terrible happened!
    precondition(outputs != nil, "Materialization failed!")
    return outputs!
  }

  /// Converts symbolic tensor inputs to concrete inputs if the associated `LazyTensorOperation`
  /// has been materialized.
  func maybeMaterializeInputs() {
    /// If `lazyTensor` is symbolic and the associated `LazyTensorOperation`
    /// has been materialized, return the corresponding concrete `LazyTensorHandle`.
    /// Otherwise, return `lazyTensor` untouched.
    func materializedAsNeeded(lazyTensor: LazyTensorHandle) -> LazyTensorHandle {
      let handle = lazyTensor.handle
      if case let .symbolic(lazyOp, index, _) = handle,
        let outputs = lazyOp.outputs
      {
        return LazyTensorHandle(_materialized: outputs[index])
      }
      return lazyTensor
    }

    /// Returns an input that is rewritten such that all symbolic values
    /// that have been materialized have been replaced by the corresponding
    /// concerete inputs. If no symbolic values have been materialized or if
    /// there are no symbolic values, return the `input` untouched.
    func materializedAsNeeded(input: Input) -> Input {
      switch input {
      case .single(let h):
        return .single(materializedAsNeeded(lazyTensor: h))
      case .list(let elements):
        return .list(elements.map { materializedAsNeeded(lazyTensor: $0) })
      }
    }
    inputs = inputs.map { materializedAsNeeded(input: $0) }
  }

  static func materialize(targets: [LazyTensorOperation]) {
    let traceInfo = LazyTensorTraceBuilder.materializationTraceInfo(targets)
    debugLog("Extracted trace:\n\(traceInfo.trace)")

    let function = TFFunction(trace: traceInfo.trace)
    debugLog("Generated TFFunction:\n\(function)")

    let allOutputs = function.execute(traceInfo.concreteInputs)

    // Slice up the outputs to various lazy tensors
    var start = 0
    for lazyOp in traceInfo.lazyOperations {
      let end = start + lazyOp.outputCount
      lazyOp.outputs = Array(allOutputs[start..<end])
      lazyOp.outputShapes = lazyOp.outputs!.map { $0.shape }
      start = end
    }
  }
}
