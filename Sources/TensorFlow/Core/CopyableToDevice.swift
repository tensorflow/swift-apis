// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

/// A type whose nested properties and elements can be copied to `Device`s.
///
/// - Note: Do not ever use this API directly. This is an implementation detail to support
///   `KeyPathIterable.move(to:)` and `KeyPathIterable.init(copying:to:)`.
///
/// - Note: this workaround is necessary because `CopyableToDevice` is a protocol with `Self`
///   requirements, so `x as? CopyableToDevice` does not work.
public protocol _CopyableToDevice {
  static func _move<Root>(
    _ root: inout Root, _ rootKeyPath: PartialKeyPath<Root>, to: Device)
}

/// A type whose nested properties and elements can be copied to a `Device`.
public protocol CopyableToDevice: _CopyableToDevice {
  /// Creates a copy of `other` on the given `Device`.
  ///
  /// All cross-device references are moved to the given `Device`.
  init(copying other: Self, to device: Device)
}

extension CopyableToDevice {
  /// Given an `inout Root` root value and a `PartialKeyPath<Root>` key path, copies the value at
  /// the key path in the root value to the given `Device`.
  ///
  /// - Note: Do not ever use this API directly. This is an implementation detail to support
  ///   `KeyPathIterable.move(to:)` and `KeyPathIterable.init(copying:to:)`.
  public static func _move<Root>(
    _ root: inout Root, _ rootKeyPath: PartialKeyPath<Root>, to device: Device
  ) {
    guard let keyPath = rootKeyPath as? WritableKeyPath<Root, Self> else {
      fatalError(
        "Failed conversion from \(rootKeyPath) to 'WritableKeyPath<\(Root.self), \(Self.self)>'"
      )
    }
    root[keyPath: keyPath] = Self.init(copying: root[keyPath: keyPath], to: device)
  }
}

extension _KeyPathIterableBase {
  /// Recursively copies all `CopyableToDevice`-conforming nested properties and elements in `root`
  /// to the given `Device` in-place.
  public func _move<Root>(
    _ root: inout Root, _ rootKeyPath: PartialKeyPath<Root>, to device: Device
  ) {
    for kp in _allKeyPathsTypeErased {
      let joinedKeyPath = rootKeyPath.appending(path: kp)!
      if let valueType = type(of: joinedKeyPath).valueType as? _CopyableToDevice.Type {
        valueType._move(&root, joinedKeyPath, to: device)
      } else if let nested = self[keyPath: kp] as? _KeyPathIterableBase {
        nested._move(&root, joinedKeyPath, to: device)
      }
    }
  }
}

extension KeyPathIterable {
  /// Recursively copies all `CopyableToDevice`-conforming nested properties and elements to the
  /// given `Device` in-place.
  public mutating func move(to device: Device) {
    _move(&self, \.self, to: device)
  }

  /// Creates a copy of `self` with all `CopyableToDevice`-conforming nested properties and elements
  /// copied to the given `Device`.
  public init(copying other: Self, to device: Device) {
    self = other
    self.move(to: device)
  }
}

extension Tensor: CopyableToDevice {
  /// Creates a copy of `other` on the given `Device`.
  public init(copying other: Tensor, to device: Device) {
    self = _Raw.toDevice(other, device)
  }
}

extension Parameter: CopyableToDevice {
  /// Creates a copy of `other` on the given `Device`.
  public convenience init(copying other: Parameter, to device: Device) {
    self.init(.init(copying: other.value, to: device))
  }
}
