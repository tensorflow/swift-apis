/// A device on which `Tensor`s can be allocated.
///
/// Currently, this is a stub because TensorFlow transfers tensors between devices on demand.
public struct Device {
    public static var `default`: Device { Device() }
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
