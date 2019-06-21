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
public func withDevice<R>(
    _ kind: DeviceKind,
    _ index: UInt = 0,
    perform body: () throws -> R
) rethrows -> R {
    return try _ExecutionContext.global.withDevice(kind, index, perform: body)
}

/// Executes a closure, making TensorFlow operations run on a device with
/// a specific name.
///
/// - Parameters:
///   - name: Device name.
///   - body: A closure whose TensorFlow operations are to be executed on the
///     specified kind of device.
///
/// Some examples of device names:
///   - "/device:CPU:0": The CPU of your machine.
///   - "/GPU:0": Short-hand notation for the first GPU of your machine that
///     is visible to TensorFlow
///   - "/job:localhost/replica:0/task:0/device:GPU:1": Fully qualified name of
///     the second GPU of your machine that is visible to TensorFlow.
public func withDevice<R>(named name: String, perform body: () throws -> R) rethrows -> R {
    return try _ExecutionContext.global.withDevice(named: name, perform: body)
}

/// Executes a closure, allowing TensorFlow to place TensorFlow operations on any device. This
/// should restore the default placement behavior.
///
/// - Parameters:
///   - body: A closure whose TensorFlow operations are to be executed on the specified kind of
///     device.
public func withDefaultDevice<R>(perform body: () throws -> R) rethrows -> R {
    return try _ExecutionContext.global.withDefaultDevice(perform: body)
}
