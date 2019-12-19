/// Represents a device where a tensor logically resides.
/// Currently, this is just a stub because tensorflow will transfer
/// tensors between devices on demand.
public struct Device {
  public static var getDefault: Device { Device() }
}

extension Tensor {
  /// The current device of a tensor.
  public var device: Device {
    @_semantics("autodiff.nonvarying")
    get {
      return Device()
    }
  }
}
