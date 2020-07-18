import CTensorFlow

#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
  import Darwin
#elseif os(Windows)
  import MSVCRT
#else
  import Glibc
#endif

@usableFromInline
internal typealias CTFSession = OpaquePointer

@usableFromInline
internal typealias CTFStatus = OpaquePointer

@usableFromInline
internal typealias CTFGraph = OpaquePointer

@usableFromInline
internal typealias CTFFunction = OpaquePointer

@usableFromInline
internal typealias CTensor = OpaquePointer

public typealias CTensorHandle = OpaquePointer

@usableFromInline
internal typealias CTFEContext = OpaquePointer

@usableFromInline
internal typealias CTFEOp = OpaquePointer

@usableFromInline
internal typealias CTFOperationDescription = OpaquePointer

@usableFromInline
internal typealias CTFETraceContext = OpaquePointer

extension UnsafeMutablePointer where Pointee == CTensorHandle? {
  @usableFromInline
  init(_ other: UnsafeMutablePointer<CTensorHandle>) {
    self.init(other._rawValue)
  }

  @usableFromInline
  init?(_ other: UnsafeMutablePointer<CTensorHandle>?) {
    guard let unwrapped = other else { return nil }
    self.init(unwrapped)
  }
}
