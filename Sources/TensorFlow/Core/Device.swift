/// A device on which `Tensor`s can be allocated.
///
/// Currently, this is a stub because TensorFlow transfers tensors between devices on demand.
public struct Device {
  public static var `default`: Device { Device() }
  public static var defaultTFEager: Device { Device() }

  /// Backend used to dispatch the tensor operations.
  public enum Backend {
    case TF_EAGER

    fileprivate var shortName: String {
      switch self {
      case .TF_EAGER: return "TF_EAGER"
      }
    }
  }
}

extension Tensor {
  /// The device on which `self` is allocated.
  public var device: Device {
    @_semantics("autodiff.nonvarying")
    get {
      return Device()
    }
  }
}

extension _Raw {
  static func toDevice<T: TensorFlowScalar>(_ x: Tensor<T>, _ device: Device) -> Tensor<T> {
    // TODO: Actually copy to device...
    return x
  }
}
