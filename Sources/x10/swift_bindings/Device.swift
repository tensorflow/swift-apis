// Copyright 2020 TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

@_implementationOnly import x10_device_wrapper

extension DeviceType {
  fileprivate var kind: Device.Kind {
    switch self {
    case CPU_DEVICE:
      return .CPU
    case GPU_DEVICE:
      return .GPU
    case TPU_DEVICE:
      return .TPU
    case REMOTE_TPU_DEVICE:
      return .REMOTE_TPU
    default:
      fatalError("Invalid device type")
    }
  }
}

extension Array where Element == Device {
  fileprivate func withDeviceList<Result>(_ body: (inout DeviceList) -> Result) -> Result {
    let cDevices: [CDevice] = map { device in
      CDevice(hw_type: device.kind.deviceType, ordinal: Int32(device.ordinal))
    }
    return cDevices.withUnsafeBufferPointer {
      (_ cDevicesBuf: UnsafeBufferPointer<CDevice>) -> Result in
      var deviceList = DeviceList(devices: cDevicesBuf.baseAddress, count: count)
      return body(&deviceList)
    }
  }
}

/// A device on which `Tensor`s can be allocated.
public struct Device {
  /// The device kind: GPU, GPU, TPU, or remote TPU.
  public let kind: Kind

  /// The device ordinal value.
  public let ordinal: Int

  /// The backend used to dispatch the tensor operations.
  public let backend: Backend

  public init(kind: Kind, ordinal: Int, backend: Backend = defaultBackend) {
    self.kind = kind
    self.ordinal = ordinal
    self.backend = backend
  }

  /// Backend used to dispatch the tensor operations.
  public enum Backend {
    case TF_EAGER
    case XLA

    fileprivate var shortName: String {
      switch self {
      case .TF_EAGER: return "TF_EAGER"
      case .XLA: return "XLA"
      }
    }

    var annotationsAvailable: String {
      switch self {
      case .TF_EAGER: return "Annotations not available in TF_EAGER."
      case .XLA: return "Annotations available in XLA."
      }
    }
  }

  /// A device kind.
  public enum Kind {
    case CPU
    case GPU
    case TPU
    case REMOTE_TPU

    fileprivate var deviceType: DeviceType {
      switch self {
      case .CPU:
        return CPU_DEVICE
      case .GPU:
        return GPU_DEVICE
      case .TPU:
        return TPU_DEVICE
      case .REMOTE_TPU:
        return REMOTE_TPU_DEVICE
      }
    }

    fileprivate var shortName: String {
      switch self {
      case .CPU: return "CPU"
      case .GPU: return "GPU"
      case .TPU: return "TPU"
      case .REMOTE_TPU: return "REMOTE_TPU"
      }
    }
  }

  var cdevice: CDevice {
    return CDevice(hw_type: kind.deviceType, ordinal: Int32(ordinal))
  }

  public var isRemote: Bool {
    return self.kind == .REMOTE_TPU
  }

  public static var defaultBackend: Backend {
    #if DEFAULT_BACKEND_EAGER
      return .TF_EAGER
    #else
      return .XLA
    #endif
  }

  /// The default `Device`.
  public static var `default`: Device {
    switch defaultBackend {
    case .TF_EAGER:
      return Device.defaultTFEager
    case .XLA:
      return Device.defaultXLA
    }
  }

  /// The default XLA device.
  public static var defaultXLA: Device {
    let defaultDevice = getDefaultDevice()
    return Device(
      kind: defaultDevice.hw_type.kind, ordinal: Int(defaultDevice.ordinal), backend: .XLA)
  }

  /// The current TF Eager device.
  public static var defaultTFEager: Device {
    // TODO: Pull this from withDevice() {} mechanism?
    return Device(kind: .CPU, ordinal: 0, backend: .TF_EAGER)
  }

  /// An array of all devices.
  public static var allDevices: [Device] {
    return deviceListToArray(DeviceListHandle(_handle: getAllDevices()))
  }

  public static func setReplicationDevices(_ devices: [Device]) {
    devices.withDeviceList { deviceList in
      x10_device_wrapper.setReplicationDevices(&deviceList)
    }
  }

  public static func getReplicationDevices() -> [Device] {
    let handle = x10_device_wrapper.getReplicationDevices()!
    return deviceListToArray(DeviceListHandle(_handle: handle))
  }

  public static func syncLiveTensorsForDevices(_ devices: [Device]) {
    devices.withDeviceList { deviceList in
      x10_device_wrapper.syncLiveTensorsForDevices(&deviceList)
    }
  }

  private static func deviceListToArray(_ deviceList: DeviceListHandle) -> [Device] {
    return (0..<deviceList.handle.pointee.count).map { i in
      let device = deviceList.handle.pointee.devices[i]
      return Device(kind: device.hw_type.kind, ordinal: Int(device.ordinal), backend: .XLA)
    }
  }

  private struct DeviceListHandle {
    init(_handle: UnsafeMutablePointer<DeviceList>) {
      handleDeleter = Handle(_handle: _handle)
    }

    var handle: UnsafeMutablePointer<DeviceList> {
      return handleDeleter.handle
    }

    // Implementation detail for deleting the pointer.
    class Handle {
      init(_handle: UnsafeMutablePointer<DeviceList>) {
        handle = _handle
      }

      deinit { destroyDeviceList(handle) }

      let handle: UnsafeMutablePointer<DeviceList>
    }

    let handleDeleter: Handle
  }
}

extension Device: Equatable {
  public static func == (lhs: Device, rhs: Device) -> Bool {
    return lhs.kind == rhs.kind && lhs.ordinal == rhs.ordinal && lhs.backend == rhs.backend
  }
}

extension Device: CustomStringConvertible {
  public var description: String {
    "Device(kind: .\(kind.shortName), ordinal: \(ordinal), backend: .\(backend.shortName))"
  }
}

extension Device {
  public var annotationsAvailable: String {
    "\(backend.annotationsAvailable)"
  }
}

extension CDevice {
  var device: Device {
    return Device(kind: hw_type.kind, ordinal: Int(ordinal), backend: .XLA)
  }
}

/// LazyTensorBarrier ensures all live tensors (on device if provided) are scheduled and running.
/// If wait is set to true, this call blocks until the computation is complete.
public func LazyTensorBarrier(on device: Device? = nil, devices: [Device] = [], wait: Bool = false)
{
  if device == Device.defaultTFEager { return }

  devices.withDeviceList { devices in
    if var cdevice = device?.cdevice {
      XLATensor_LazyTensorBarrier(&cdevice, &devices, wait)
    } else {
      XLATensor_LazyTensorBarrier(nil, &devices, wait)
    }
  }
}
