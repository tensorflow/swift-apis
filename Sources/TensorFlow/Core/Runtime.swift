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

// This file defines the Swift runtime support for TensorFlow computation.
//
// This file should only contain internal details: runtime-related public APIs
// should be defined in `Execution.swift`.
//
// Design notes on TF eager based runtime:
//
// A global context (`_ExecutionContext.global`) is used to manage all tensor
// computation and transfers.
//
// Potential TODOs:
// - Support async on platforms other than Linux and FreeBSD.
// - Revisit the concurrency model and see if Dispatch can be built without
//   Foundation.
//
//===----------------------------------------------------------------------===//

import CTensorFlow

#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
  import Darwin
#elseif os(Windows)
  import ucrt
#else
  import Glibc
#endif

#if os(Windows)
  // NOTE: although the function is racy, we do not really care as the
  // usage here is to not override the value if the user specified one before
  // creating the process.
  @discardableResult
  func setenv(_ variable: String, _ value: String, _ `override`: Int) -> Int {
    guard `override` > 0 || getenv(variable) == nil else { return 0 }
    return Int(_putenv_s(variable, value))
  }
#endif

/// The configuration for the compiler runtime.
// TODO(hongm): Revisit the longer-term design.
// @_frozen // SR-9739
public enum _RuntimeConfig {
  // TODO: change this and subsequent properties from static to thread local.
  /// When false, tensorflow runtime will be initialized before running any tensor program in this
  /// process.
  static public var tensorFlowRuntimeInitialized = false

  /// When true, let TensorFlow GPU memory allocation start small and grow as needed. Otherwise,
  /// The entire GPU memory region is pre-allocated.
  static public var gpuMemoryAllowGrowth = true

  /// The number of CPU devices.
  static public var cpuDeviceCount: UInt32 = 1

  /// Specifies whether the TensorFlow computation runs in a local (in-process) session, or a
  /// remote session with the specified server definition.
  // @_frozen // SR-9739
  public enum RuntimeSession {
    case local
    case remote(serverDef: String)
  }
  static public var session: RuntimeSession = .local

  /// When true, use lazy evaluation.
  static public var useLazyTensor: Bool = false

  /// When true, prints various debug messages on the runtime state.
  ///
  /// If the value is true when running tensor computation for the first time in the process, INFO
  /// log from TensorFlow will also get printed.
  static public var printsDebugLog = false

  /// Specifies the verbose log level in TensorFlow; a higher level prints out more log. Only
  /// meaningful when `printsDebugLog` is true, and must be within [0, 4] in that case.
  static public var tensorflowVerboseLogLevel: Int32 = 0 {
    willSet {
      debugLog("About to set tensorflowVerboseLogLevel to \(newValue)")
      guard newValue >= 0 && newValue <= 4 else {
        fatalError("Invalid tensorflowVerboseLogLevel value \(newValue)")
      }
    }
  }
}

private func configureRuntimeFromEnvironment() {
  if let value = getenv("SWIFT_TENSORFLOW_ENABLE_DEBUG_LOGGING"),
    String(cString: value).lowercased() == "true"
  {
    _RuntimeConfig.printsDebugLog = true
    debugLog("Turning on debug logging from env.")
  }

  if let value = getenv("SWIFT_TENSORFLOW_ENABLE_LAZY_TENSOR"),
    String(cString: value).lowercased() == "true"
  {
    _RuntimeConfig.useLazyTensor = true
    debugLog("Turning on lazy tensor from env.")
  }

  if let value = getenv("SWIFT_TENSORFLOW_VERBOSE_LOG_LEVEL") {
    guard var verboseLevel = Int32(String(cString: value)) else {
      fatalError("SWIFT_TENSORFLOW_VERBOSE_LOG_LEVEL must take an int value.")
    }
    if verboseLevel > 4 {
      verboseLevel = 4
    }
    _RuntimeConfig.tensorflowVerboseLogLevel = verboseLevel
    debugLog("Setting TF logging verbose level to \(verboseLevel) from env.")
  }

  if let value = getenv("SWIFT_TENSORFLOW_SERVER_ADDRESS") {
    let address = String(cString: value)
    debugLog("Env var SWIFT_TENSORFLOW_SERVER_ADDRESS has value \(address).")
    if address == "local" {
      _RuntimeConfig.session = .local
      debugLog("Using local TF session.")
    } else {
      guard let idx = address.firstIndex(of: ":"),
        let endIdx = address.index(idx, offsetBy: 3, limitedBy: address.endIndex),
        address[idx..<endIdx] == "://"
      else {
        fatalError("SWIFT_TENSORFLOW_SERVER_ADDRESS must start with 'grpc://'.")
      }

      let `protocol` = address[address.startIndex..<idx]
      let target = address[endIdx..<address.endIndex]
      _RuntimeConfig.session = .remote(
        serverDef: """
          cluster {
          job {
              name: "localhost"
              tasks {
              key: 0
              value: "127.0.0.1:0"
              }
              tasks {
              key: 1
              value: "\(target)"
              }
          }
          }
          job_name: "localhost"
          task_index: 0
          protocol: "\(`protocol`)"
          """)
      debugLog("Setting TF server address to \(address) from env.")

      // At the moment, without TF_EAGER_REMOTE_USE_SEND_TENSOR_RPC=1, running on TPUs freezes.
      // Therefore, we set this environment variable to 1 unless it's set explicitly.
      if let value = getenv("TF_EAGER_REMOTE_USE_SEND_TENSOR_RPC") {
        debugLog("TF_EAGER_REMOTE_USE_SEND_TENSOR_RPC already set:")
        debugLog(String(cString: value))
      } else {
        setenv("TF_EAGER_REMOTE_USE_SEND_TENSOR_RPC", "1", /*override*/ 0)
        debugLog("Setting TF_EAGER_REMOTE_USE_SEND_TENSOR_RPC to 1")
      }
    }
  }

  if let value = getenv("SWIFT_TENSORFLOW_CPU_DEVICE_COUNT") {
    guard let cpuDeviceCount = UInt32(String(cString: value)) else {
      fatalError("SWIFT_TENSORFLOW_CPU_DEVICE_COUNT must take an int value.")
    }
    _RuntimeConfig.cpuDeviceCount = cpuDeviceCount
    debugLog("Setting number of CPU devices to \(cpuDeviceCount) from env.")
  }
}

/// The host of any tensor computation.
public final class _ExecutionContext {
  /// Global context storing all available devices, loaded functions, etc.
  public static let global: _ExecutionContext = _ExecutionContext()

  /// List of devices available to this execution context.
  /// Devices are represented by their names in TensorFlow notation.
  /// See documentation for `withDevice(named:perform:)` to learn about device names.
  public private(set) var deviceNames: [String] = []

  /// The buffer storing a serialized TensorFlow config proto.
  public let tensorFlowConfig: UnsafeMutablePointer<TF_Buffer>

  /// The TFE_Context object.
  @usableFromInline let eagerContext: CTFEContext

  /// The status for checking TensorFlow errors.
  @usableFromInline let status: CTFStatus = TF_NewStatus()

  /// The mutex for preventing potential concurrent access.
  private var mutex: Mutex = Mutex()

  /// Initializes a new execution context by initializing available devices.
  @usableFromInline
  init() {
    configureRuntimeFromEnvironment()

    // Suppress TensorFlow logging, unless the user specified a log level.
    setenv("TF_CPP_MIN_LOG_LEVEL", "3", /*override*/ 0)

    debugLog("Initializing global context.")

    // Initialize the TF runtime exactly once. Only affects local execution
    // (when _RuntimeConfig.tensorFlowServer is set to "").
    if !_RuntimeConfig.tensorFlowRuntimeInitialized {
      // Install a signal handler to ensure we exit when interrupted.
      signal(SIGINT) { _ in
        print("Caught interrupt signal, exiting...")
        exit(1)
      }

      var args = ["dummyProgramName"]
      if _RuntimeConfig.printsDebugLog {
        args.append("--alsologtostderr")
      }
      if _RuntimeConfig.tensorflowVerboseLogLevel > 0 {
        args.append("--v=\(_RuntimeConfig.tensorflowVerboseLogLevel)")
      }
      // Collect all the strings' utf8 bytes into a single array so that we can
      // address all the strings with a single `flattenedStringBytes.withUnsafeBufferPointer`.
      var flattenedStringBytes: [Int8] = []
      var lengths: [Int] = []
      for arg in args {
        let bytes = arg.utf8CString
        flattenedStringBytes.append(contentsOf: bytes)
        lengths.append(bytes.count)
      }

      // Calculate the addresses of all the strings within our single buffer, and then call
      // TF_InitMain.
      flattenedStringBytes.withUnsafeMutableBufferPointer { buffer in
        var stringAddrs: [UnsafeMutablePointer<Int8>?] = []
        var currentStringAddr = buffer.baseAddress
          .map(UnsafeMutablePointer.init)
        for length in lengths {
          stringAddrs.append(currentStringAddr)
          currentStringAddr = currentStringAddr?.advanced(by: length)
        }

        stringAddrs.withUnsafeMutableBufferPointer { stringAddrsBuffer in
          #if !USING_X10_BACKEND
            var cArgsCount = Int32(args.count)
            var cArgs = stringAddrsBuffer.baseAddress.map(UnsafeMutablePointer.init)
            TF_InitMain(nil, &cArgsCount, &cArgs)
          #endif
        }
      }
      _RuntimeConfig.tensorFlowRuntimeInitialized = true
    }

    guard let opts = TFE_NewContextOptions() else {
      fatalError("ContextOptions object can never be nil.")
    }

    // Create TF config object.
    if _RuntimeConfig.gpuMemoryAllowGrowth {
      debugLog("Allowing growth for GPU memory allocator.")
    }
    self.tensorFlowConfig = TF_CreateConfig(
      /* enable_xla_compilation */0,
      _RuntimeConfig.gpuMemoryAllowGrowth ? 1 : 0,
      _RuntimeConfig.cpuDeviceCount)
    TFE_ContextOptionsSetConfig(
      opts,
      tensorFlowConfig.pointee.data,
      tensorFlowConfig.pointee.length,
      status)
    checkOk(status)

    let ctx = TFE_NewContext(opts, status)
    checkOk(status)
    self.eagerContext = ctx!
    TFE_DeleteContextOptions(opts)
    checkOk(status)

    #if !os(Windows)
    if case .remote(let serverDef) = _RuntimeConfig.session {
      debugLog("Setting up the server def to \(serverDef)...")	
      serverDef.utf8CString.withUnsafeBufferPointer { ptr in
        TFE_ContextSetServerDef(
          eagerContext, /*keep_alive_secs*/ 0, ptr.baseAddress,
          serverDef.utf8CString.count, status)
        checkOk(status)
      }
    }
    #endif

    let devices = TFE_ContextListDevices(eagerContext, status)
    checkOk(status)
    defer { TF_DeleteDeviceList(devices!) }

    let deviceCount = TF_DeviceListCount(devices!)
    debugLog("There are \(deviceCount) devices.")
    for deviceId in 0..<deviceCount {
      let cDeviceName = TF_DeviceListName(devices, deviceId, status)
      checkOk(status)
      let deviceName = String(cString: cDeviceName!)
      let cDeviceType = TF_DeviceListType(devices, deviceId, status)
      checkOk(status)
      let deviceType = String(cString: cDeviceType!)
      debugLog("Device \(deviceId) has type \(deviceType) and name \(deviceName).")
      deviceNames.append(deviceName)
    }
  }

  deinit {
    debugLog("De-initializing global context.")
    // Delete all loaded programs.
    TFE_DeleteContext(eagerContext)
    TF_DeleteBuffer(tensorFlowConfig)
    TF_DeleteStatus(status)
  }
}

@available(
  *, deprecated, message: "makeOp will go away in favor of directly dispatching custom ops."
)
public func _makeOp(_ name: String, _ nOutputs: Int) -> TFTensorOperation {
  _ExecutionContext.makeOp(name, nOutputs)
}

extension _ExecutionContext {
  // The execution mode is effectively encoded in the TensorOperation.
  // We can use this to switch between different execution modes.
  // TODO: Can we interop between modes?
  @usableFromInline
  static func makeOp(
    _ name: String, _ outputCount: Int
  ) -> TFTensorOperation {
    return _ThreadLocalState.useLazyTensor
      ? LazyTensorOperation(name, outputCount)
      : TFE_Op(name, outputCount)
  }
}

internal func _trace<In: TensorGroup, Out: TensorGroup>(_ fn: (In) -> Out) -> TFFunction {
  let useLazyTensor = _ThreadLocalState.useLazyTensor
  defer { _ThreadLocalState.useLazyTensor = useLazyTensor }
  _ThreadLocalState.useLazyTensor = true
  let trace = LazyTensorTraceBuilder.trace(fn)
  return TFFunction(trace: trace)
}

// Trace the given function to generate a TF graph and return a closure that can be used to launch
// the graph.
public func _graph<In: TensorGroup, Out: TensorGroup>(
  _ fn: (In) -> Out,
  useXLA: Bool = false
) -> (In) -> Out {
  let tffunc = _trace(fn)
  return { input in
    let inputHandles = input._tensorHandles.map { $0._tfeTensorHandle }
    let outputHandles = tffunc.execute(inputHandles, usingXLA: useXLA)
    return Out(_handles: outputHandles)
  }
}

/// Trace the given function and return the name of the corresponding `TF_Function: In -> Out` that
/// was created.
public func _tffunc<In: TensorGroup, Out: TensorGroup>(_ fn: (In) -> Out) -> String {
  let tffunc = _trace(fn)
  return tffunc.name
}

extension _ExecutionContext {
  /// Returns a valid TensorFlow device name, which corresponds to the closest enclosing call to
  /// one of the overloads of withDevice. A return value of `nil` indicates the absence of a
  /// withDevice call on the call stack or the presence of an immediately enclosing
  /// `withDefaultDevice(perform)` call.
  var currentDeviceName: String? {
    return _ThreadLocalState.local.deviceScopes._currentDevice
  }

  /// See documentation for the top-level `withDevice(_:_:perform)`.
  func withDevice<R>(
    _ kind: DeviceKind,
    _ index: UInt = 0,
    perform body: () throws -> R
  ) rethrows -> R {
    let name: String
    switch kind {
    case .cpu:
      name = "/job:localhost/replica:0/task:0/device:CPU:\(index)"
    case .gpu:
      name = "/job:localhost/replica:0/task:0/device:GPU:\(index)"
    case .tpu:
      // According to server def generated when you set
      // SWIFT_TENSORFLOW_SERVER_ADDRESS, the TPUs will all be on task 1.
      name = "/job:localhost/replica:0/task:1/device:TPU:\(index)"
    }
    return try withDevice(named: name, perform: body)
  }

  /// See documentation for the top-level `withDevice(named:perform)`.
  func withDevice<R>(named name: String, perform body: () throws -> R) rethrows -> R {
    guard deviceNames.contains(name) else {
      fatalError("Device \(name) not found")
    }
    _ThreadLocalState.local.deviceScopes.pushDevice(name)
    let result = try body()
    _ThreadLocalState.local.deviceScopes.popDevice()
    return result
  }

  /// See documentation for the top-level `withDefaultDevice(perform)`.
  func withDefaultDevice<R>(perform body: () throws -> R) rethrows -> R {
    _ThreadLocalState.local.deviceScopes.pushDevice(nil)
    let result = try body()
    _ThreadLocalState.local.deviceScopes.popDevice()
    return result
  }
}

extension _ExecutionContext {
  /// Synchronously execute the body, preventing asynchronous computation from corrupting the
  /// context data.
  private func sync<Result>(execute body: () throws -> Result) rethrows -> Result {
    let lockStatus = mutex.acquire()
    internalConsistencyCheck(lockStatus == 0)
    defer {
      let unlockStatus = mutex.release()
      internalConsistencyCheck(unlockStatus == 0)
    }
    return try body()
  }
}

@inlinable
func _TFCEagerExecute(
  _ op: CTFEOp,
  _ retvals: UnsafeMutablePointer<OpaquePointer?>,
  _ retvalCount: UnsafeMutablePointer<Int32>,
  _ status: CTFStatus
) {
  TFE_Execute(op, retvals, retvalCount, status)
}

//===----------------------------------------------------------------------===//
// - MARK: Dynamic compilation (per-op dispatch) entrypoints
//===----------------------------------------------------------------------===//

@usableFromInline
func _TFCGetGlobalEagerContext() -> CTFEContext {
  debugLog("Calling _GetGlobalEagerContext()")
  return _ExecutionContext.global.eagerContext
}

/// Adds `handle` as an input to `op`.
@usableFromInline
func _TFCOpAddInputFromTensorHandle(_ op: CTFEOp, _ handle: _AnyTensorHandle, _ status: CTFStatus) {
  TFE_OpAddInput(op, handle._cTensorHandle, status)
}

/// Adds `t` as an input or inputs to `op`. Returns the number of inputs added.
@usableFromInline
func _TFCOpAddInputFromTensorGroup<T: TensorArrayProtocol>(
  _ op: CTFEOp,
  _ t: T,
  _ status: CTFStatus
) -> Int32 {
  let count = t._tensorHandleCount
  let buffer = UnsafeMutableBufferPointer<CTensorHandle>.allocate(capacity: Int(count))
  defer { buffer.deallocate() }
  t._unpackTensorHandles(into: buffer.baseAddress)
  for handle in buffer {
    TFE_OpAddInput(op, handle, status)
    guard TF_GetCode(status) == TF_OK else {
      return 0
    }
  }
  return count
}

@usableFromInline
func _TFCOpAddInputFromAnyTensors(_ op: CTFEOp, _ tensors: [AnyTensor], _ status: CTFStatus) {
  for tensor in tensors {
    let handle = tensor._rawTensorHandle
    TFE_OpAddInput(op, handle, status)
    checkOk(status)
  }
}

// _TFCOpSetAttr*Array functions are wrappers around TFE_OpSetAttr*List functions. The wrappers
// handle converting the Swift Stdlib Array<T> values into buffers that TFE_OpSetAttr*List functions
// can read.

@usableFromInline
func _TFCOpSetAttrTypeArray(
  _ op: CTFEOp,
  _ attrName: UnsafePointer<Int8>,
  _ value: [TensorDataType]
) {
  value.withUnsafeBufferPointer { buffer in
    buffer.withMemoryRebound(to: TF_DataType.self) { reboundBuffer in
      TFE_OpSetAttrTypeList(
        op, attrName, reboundBuffer.baseAddress, Int32(reboundBuffer.count))
    }
  }
}

/// A class to keep around thread local state:
///  - DeviceScopes
///  - LazyTensorContext
class _ThreadLocalState {
  var deviceScopes = DeviceScopes()

  var lazyTensorContext = LazyTensorContext()

  static var useLazyTensor: Bool {
    get {
      _ThreadLocalState.local.lazyTensorEnabled ?? _RuntimeConfig.useLazyTensor
    }
    set {
      _ThreadLocalState.local.lazyTensorEnabled = newValue
    }
  }

  /// When true, use lazy evaluation. If this is not set, we should use the
  /// value of `_RuntimeConfig.useLazyTensor` to determine if lazy evaluation
  /// is enabled.
  private var lazyTensorEnabled: Bool? = nil

  private static let key: ThreadLocalStorage.Key =
    ThreadLocalStorage.Key {
      #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
        Unmanaged<AnyObject>.fromOpaque($0).release()
      #else
        Unmanaged<AnyObject>.fromOpaque($0!).release()
      #endif
    }

  @usableFromInline
  static var local: _ThreadLocalState {
    if let state = ThreadLocalStorage.get(for: key) {
      return Unmanaged.fromOpaque(state).takeUnretainedValue()
    }

    let state = _ThreadLocalState()
    ThreadLocalStorage.set(
      value: Unmanaged.passRetained(state).toOpaque(),
      for: key)
    return state
  }
}

/// Stack of devices that models nested calls to withDevice/withDefaultDevice. Devices are
/// represented by their names in TensorFlow notation. See documentation for
/// `withDevice(named:perform:)` to learn about device names.
///
/// All TensorFlow operations will be put on the topmost device on the stack. When the stack is
/// empty or the topmost device is `nil`, that allows TensorFlow to place operations on any device
/// that it sees fit.
@usableFromInline
struct DeviceScopes {
  var deviceStack: [String?] = []

  var _currentDevice: String? {
    return deviceStack.last ?? nil
  }

  @usableFromInline
  mutating func pushDevice(_ device: String?) {
    deviceStack.append(device)
  }

  @usableFromInline
  mutating func popDevice() {
    internalConsistencyCheck(deviceStack.popLast() != nil)
  }
}

#if !USING_X10_BACKEND
  // Evaluate the pullback on a one.
  @usableFromInline
  func pullbackOfOneLikeY<T: TensorFlowFloatingPoint, R>(
    y: Tensor<T>,
    pullback: (Tensor<T>) -> R
  ) -> R {
    pullback(Tensor<T>(1))
  }
#endif

@usableFromInline
func _TFCOpSetDeviceFromScope(_ op: CTFEOp, _ status: CTFStatus) {
  if let deviceName = _ExecutionContext.global.currentDeviceName {
    TFE_OpSetDevice(op, deviceName, status)
  }
}
