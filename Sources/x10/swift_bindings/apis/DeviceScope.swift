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

/// Keeps around the current device to place AD zero tensors until AD can switch over to using
/// instance zeros.
class _DeviceThreadLocalState {
  var deviceStack: [Device] = []

  var currentDevice: Device { return deviceStack.last ?? .default }

  var isReducedPrecision: Bool = false

  private static let key: ThreadLocalStorage.Key =
    ThreadLocalStorage.Key {
      #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
        let _: AnyObject = Unmanaged.fromOpaque($0).takeRetainedValue()
      #else
        let _: AnyObject = Unmanaged.fromOpaque($0!).takeRetainedValue()
      #endif
    }

  @usableFromInline
  static var local: _DeviceThreadLocalState {
    if let state = ThreadLocalStorage.get(for: key) {
      return Unmanaged.fromOpaque(state).takeUnretainedValue()
    }

    let state = _DeviceThreadLocalState()
    ThreadLocalStorage.set(
      value: Unmanaged.passRetained(state).toOpaque(),
      for: key)
    return state
  }
}

// Evaluate the pullback on a one with the same device and precision as y.
@usableFromInline
func pullbackOfOneLikeY<T: TensorFlowFloatingPoint, R>(
  y: Tensor<T>,
  pullback: (Tensor<T>) -> R
) -> R {
  let adDevice = y.device
  _DeviceThreadLocalState.local.deviceStack.append(adDevice)
  let savedPrecision = _DeviceThreadLocalState.local.isReducedPrecision
  _DeviceThreadLocalState.local.isReducedPrecision = y.isReducedPrecision
  let result = pullback(Tensor<T>(1, deviceAndPrecisionLike: y))
  _DeviceThreadLocalState.local.isReducedPrecision = savedPrecision
  precondition(_DeviceThreadLocalState.local.deviceStack.popLast() != nil)
  return result
}
