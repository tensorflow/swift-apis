public enum DeviceKind {
  case cpu
  case gpu
  case tpu
}

public func withDevice<R>(
  _ kind: DeviceKind,
  _ index: UInt = 0,
  perform body: () throws -> R
) rethrows -> R {
  fatalError()
}

public func withDevice<R>(named name: String, perform body: () throws -> R) rethrows -> R {
  fatalError()
}

public func withDefaultDevice<R>(perform body: () throws -> R) rethrows -> R {
  fatalError()
}
