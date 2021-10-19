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

import _Differentiation

@_implementationOnly import x10_xla_tensor_wrapper

/// A type whose nested floating-point tensor properties and elements can be converted from full
/// precision to reduced precision and vice versa.
///
/// - Note: Do not ever use this API directly. This is an implementation detail to support
///   `KeyPathIterable.convertToReducedPrecision` and `KeyPathIterable.convertToFullPrecision`.
///
/// - Note: this workaround is necessary because `ReducedPrecisionConvertible` is a protocol with
///   `Self` requirements, so `x as? ReducedPrecisionConvertible` does not work.
public protocol _ReducedPrecisionConvertible {
  /// Given an `inout Root` root value and a `PartialKeyPath<Root>` key path, converts the value at
  /// the key path in the root value to reduced precision.
  static func _convertToReducedPrecision<Root>(
    _ root: inout Root, _ rootKeyPath: PartialKeyPath<Root>)

  /// Given an `inout Root` root value and a `PartialKeyPath<Root>` key path, converts the value at
  /// the key path in the root value to full precision.
  static func _convertToFullPrecision<Root>(
    _ root: inout Root, _ rootKeyPath: PartialKeyPath<Root>)
}

/// A type whose nested floating-point tensor properties and elements can be converted from full
/// precision to reduced precision and vice versa.
public protocol ReducedPrecisionConvertible: _ReducedPrecisionConvertible {
  /// Returns a copy of `self`, converting nested floating-point tensor properties and elements
  /// from full precision to reduced precision.
  var toReducedPrecision: Self { get }

  /// Returns a copy of `self`, converting nested floating-point tensor properties and elements
  /// from full precision to reduced precision.
  var toFullPrecision: Self { get }
}

extension ReducedPrecisionConvertible {
  /// Given an `inout Root` root value and a `PartialKeyPath<Root>` key path, converts the physical
  /// scalar type of the value at the key path in the root value to `BFloat16`.
  ///
  /// - Note: Do not ever use this API directly. This is an implementation detail to support
  ///   `KeyPathIterable.convertToReducedPrecision` and `KeyPathIterable.convertToFullPrecision`.
  public static func _convertToReducedPrecision<Root>(
    _ root: inout Root, _ rootKeyPath: PartialKeyPath<Root>
  ) {
    guard let keyPath = rootKeyPath as? WritableKeyPath<Root, Self> else {
      fatalError(
        "Failed conversion from \(rootKeyPath) to 'WritableKeyPath<\(Root.self), \(Self.self)>'")
    }
    root[keyPath: keyPath] = root[keyPath: keyPath].toReducedPrecision
  }

  /// Given an `inout Root` root value and a `PartialKeyPath<Root>` key path, converts the physical
  /// scalar type of the value at the key path in the root value from `BFloat16` to a different
  /// floating-point type.
  ///
  /// - Note: Do not ever use this API directly. This is an implementation detail to support
  ///   `KeyPathIterable.convertToReducedPrecision` and `KeyPathIterable.convertToFullPrecision`.
  public static func _convertToFullPrecision<Root>(
    _ root: inout Root, _ rootKeyPath: PartialKeyPath<Root>
  ) {
    guard let keyPath = rootKeyPath as? WritableKeyPath<Root, Self> else {
      fatalError(
        "Failed conversion from \(rootKeyPath) to 'WritableKeyPath<\(Root.self), \(Self.self)>'")
    }
    root[keyPath: keyPath] = root[keyPath: keyPath].toFullPrecision
  }
}

extension _KeyPathIterableBase {
  /// Recursively converts all `_ReducedPrecisionConvertible`-conforming nested properties and
  /// elements in `root` to reduced precision.
  public func _convertToReducedPrecision<Root>(
    _ root: inout Root, _ rootKeyPath: PartialKeyPath<Root>
  ) {
    for kp in _allKeyPathsTypeErased {
      let joinedKeyPath = rootKeyPath.appending(path: kp)!
      if let valueType = type(of: joinedKeyPath).valueType as? _ReducedPrecisionConvertible.Type {
        valueType._convertToReducedPrecision(&root, joinedKeyPath)
      } else if let value = self[keyPath: kp], let nested = value as? _KeyPathIterableBase {
        nested._convertToReducedPrecision(&root, joinedKeyPath)
      }
    }
  }

  /// Recursively converts all `_ReducedPrecisionConvertible`-conforming nested properties and
  /// elements in `root` to full precision.
  public func _convertToFullPrecision<Root>(
    _ root: inout Root, _ rootKeyPath: PartialKeyPath<Root>
  ) {
    for kp in _allKeyPathsTypeErased {
      let joinedKeyPath = rootKeyPath.appending(path: kp)!
      if let valueType = type(of: joinedKeyPath).valueType as? _ReducedPrecisionConvertible.Type {
        valueType._convertToFullPrecision(&root, joinedKeyPath)
      } else if let value = self[keyPath: kp], let nested = value as? _KeyPathIterableBase {
        nested._convertToFullPrecision(&root, joinedKeyPath)
      }
    }
  }
}

extension KeyPathIterable {
  /// Recursively converts all `_ReducedPrecisionConvertible`-conforming nested properties and elements
  /// to reduced precision.
  public mutating func convertToReducedPrecision() {
    _convertToReducedPrecision(&self, \.self)
  }

  /// Recursively converts all `_ReducedPrecisionConvertible`-conforming nested properties and elements
  /// to full precision.
  public mutating func convertToFullPrecision() {
    _convertToFullPrecision(&self, \.self)
  }

  /// Returns a copy of `self`, converting all `_ReducedPrecisionConvertible`-conforming nested
  /// properties and elements to reduced precision.
  public var toReducedPrecision: Self {
    var result = self
    result.convertToReducedPrecision()
    return result
  }

  /// Returns a copy of `self`, converting all `_ReducedPrecisionConvertible`-conforming nested
  /// properties and elements to full precision.
  public var toFullPrecision: Self {
    var result = self
    result.convertToFullPrecision()
    return result
  }
}

extension Tensor {
  /// Returns true if the physical scalar type is reduced precision.
  ///
  /// Currently, reduced precision physical scalar types include only `BFloat16`.
  public var isReducedPrecision: Bool {
    return device.backend == .XLA && xlaTensor.physicalScalarType == XLATensorScalarType_BFloat16
  }

  /// Promotes a scalar to a tensor with the same device and precision as the given tensor.
  // TODO (SR-12968): Mark `tensor` with `@noDerivative` and remove custom vjp below.
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public init(_ value: Scalar, deviceAndPrecisionLike tensor: Tensor) {
    let device = tensor.device
    let tmp = Tensor(value, on: device)
    self = tensor.isReducedPrecision ? tmp.toReducedPrecision : tmp
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  // TODO (SR-12968): Remove when `tensor` can be marked `@noDerivative` in `init`.
  // This currently places the pullback results of `tensor` on the correct device.
  @usableFromInline
  @derivative(of: init(_:deviceAndPrecisionLike:))
  static func vjpInitDeviceAndPrecisionLike(
    _ value: Scalar,
    deviceAndPrecisionLike tensor: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> (Scalar, Tensor)) {
    // Get device and precision in forward pass to avoid capturing `tensor` in pullback.
    let device = tensor.device
    let useReducedPrecision = tensor.isReducedPrecision
    let result = Tensor(value, on: device)
    return (useReducedPrecision ? result.toReducedPrecision : result, {
      let tmp = Tensor(0, on: device)
      return ($0.scalarized(), useReducedPrecision ? tmp.toReducedPrecision : tmp)
    })
  }
}

extension Tensor: ReducedPrecisionConvertible, _ReducedPrecisionConvertible {
  /// Returns a copy of `self` converted to `BFloat16` physical scalar type.
  public var toReducedPrecision: Self {
    if isReducedPrecision {
      fatalError("Must not already have reduced precision")
    }
    if Scalar.self != Float.self {
      fatalError("Reduced precision is only supported for Float tensors")
    }
    return _Raw.physicalCast(self, destType: BFloat16.self)
  }

  /// Returns a copy of `self` converted to `Scalar` physical scalar type.
  public var toFullPrecision: Self {
    if !isReducedPrecision {
      fatalError("Must have reduced precision")
    }
    if Scalar.self != Float.self {
      fatalError("Reduced precision is only supported for Float tensors")
    }
    return _Raw.physicalCast(self, destType: Scalar.self)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @usableFromInline
  @derivative(of: toReducedPrecision)
  func _vjpToReducedPrecision() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (toReducedPrecision, { $0.toFullPrecision })
  }

  @usableFromInline
  @derivative(of: toFullPrecision)
  func _vjpToFullPrecision() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (toFullPrecision, { $0.toReducedPrecision })
  }
}
