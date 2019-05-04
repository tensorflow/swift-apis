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

// If `serverAddress` is nil, use local session (good for forge testing).
//
// FIXME: We need transparent here because deabstraction isn't inlining this function.  We need to 
// inline if a callee contains tensor ops, not only if it takes and returns a TensorFlow value.
@_transparent
@available(*, deprecated)
public func enableTPU(serverAddress: String? = nil, infeed: Bool = true) {
    _RuntimeConfig.executionMode = .tpu
    if let serverAddress = serverAddress {
        _RuntimeConfig.session = .remote(serverDef: serverAddress)
    }
}

// FIXME: Extend the interface to support multiple GPU devices, and unify it with enableTPU() above.
@_transparent
@available(*, deprecated)
public func enableGPU() {
}

@_transparent
@available(*, deprecated)
public func enableCPU() {
}

/// A TensorFlow device kind.
public enum DeviceKind {
    /// The CPU device kind.
    case cpu
    /// The GPU device kind.
    case gpu
    /// The TPU device kind.
    case tpu
}

/// Executes a closure, making TensorFlow operations run on a specific kind of device.
///
/// - Parameters:
///   - kind: A kind of device to run TensorFlow operations on.
///   - index: The device to run the ops on.
///   - body: A closure whose TensorFlow operations are to be executed on the
///     specified kind of device.
// Use `@inline(never)` to ensure correctness in scoped device placement. See
// https://bugs.swift.org/browse/SR-9535 for more context.
@inline(never)
public func withDevice<R>(
    _ kind: DeviceKind,
    _ index: UInt = 0,
    perform body: () throws -> R
) rethrows -> R {
    _ThreadLocalState.value.pushDevice((kind, index))
    let result = try body()
    _ThreadLocalState.value.popDevice()
    return result
}

/// Executes a closure, allowing TensorFlow to place TensorFlow operations on any device. This
/// should restore the default placement behavior.
///
/// - Parameters:
///   - body: A closure whose TensorFlow operations are to be executed on the specified kind of 
///     device.
@inline(never)
public func withDefaultDevice<R>(perform body: () throws -> R) rethrows -> R {
    _ThreadLocalState.value.pushDevice(nil)
    let result = try body()
    _ThreadLocalState.value.popDevice()
    return result
}
