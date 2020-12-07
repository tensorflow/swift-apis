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

#if TENSORFLOW_USE_STANDARD_TOOLCHAIN
@_spi(Reflection) import Swift

/// Implementation detail for reflection.
///
/// This should contain the methods of `VectorProtocol`
/// that do not require Self constraints.
public protocol _VectorProtocol {
  typealias VectorSpaceScalar = Float

  /// Adds the specified scalar to `self`.
  mutating func add(_ x: VectorSpaceScalar)

  /// Subtracts the specified scalar to `self`.
  mutating func subtract(_ x: VectorSpaceScalar)

  /// Scales `self` by the specified scalar.
  mutating func scale(by scalar: VectorSpaceScalar)
}

extension VectorProtocol {
  internal static func visitChildren(
    _ body: (PartialKeyPath<Self>, _VectorProtocol.Type) -> Void
  ) {
    if !_forEachFieldWithKeyPath(
      of: Self.self,
      body: { name, kp in
        let valueType = type(of: kp).valueType
        guard let valueType = valueType as? _VectorProtocol.Type else {
          fatalError("not VectorProtocol: \(valueType)")
        }
        body(kp, valueType)
        return true
      })
    {
      fatalError("Unreflectable member of \(Self.self) while implementing VectorProtocol.")
    }
  }
}

extension _VectorProtocol {
  static func add<Root>(_ v: inout Root, _ kp: PartialKeyPath<Root>, _ x: VectorSpaceScalar) {
    v[keyPath: (kp as! WritableKeyPath<Root, Self>)].add(x)
  }
  static func subtract<Root>(_ v: inout Root, _ kp: PartialKeyPath<Root>, _ x: VectorSpaceScalar)
  {
    v[keyPath: (kp as! WritableKeyPath<Root, Self>)].subtract(x)
  }
  static func scale<Root>(
    _ v: inout Root, _ kp: PartialKeyPath<Root>, by scalar: VectorSpaceScalar
  ) {
    v[keyPath: (kp as! WritableKeyPath<Root, Self>)].scale(by: scalar)
  }
}

/// A type that represents an unranked vector space. Values of this type are
/// elements in this vector space and have either no shape or a static shape.
public protocol VectorProtocol: _VectorProtocol & AdditiveArithmetic {
  /// The type of scalars in the vector space.
  associatedtype VectorSpaceScalar = Float

  func adding(_ x: VectorSpaceScalar) -> Self

  mutating func add(_ x: VectorSpaceScalar)

  func subtracting(_ x: VectorSpaceScalar) -> Self

  mutating func subtract(_ x: VectorSpaceScalar)

  /// Returns `self` multiplied by the given scalar.
  func scaled(by scalar: VectorSpaceScalar) -> Self

  /// Multiplies `self` by the given scalar.
  mutating func scale(by scalar: VectorSpaceScalar)
}

extension VectorProtocol {
  public mutating func add(_ x: VectorSpaceScalar) {
    self = adding(x)
  }

  public mutating func subtract(_ x: VectorSpaceScalar) {
    self = subtracting(x)
  }

  public mutating func scale(by scalar: VectorSpaceScalar) {
    self = scaled(by: scalar)
  }
}

extension VectorProtocol {
  public func adding(_ x: VectorSpaceScalar) -> Self {
    var out = self
    Self.visitChildren { kp, t in t.add(&out, kp, x) }
    return out
  }
  public func subtracting(_ x: VectorSpaceScalar) -> Self {
    var out = self
    Self.visitChildren { kp, t in t.subtract(&out, kp, x) }
    return out
  }
  public func scaled(by scalar: VectorSpaceScalar) -> Self {
    var out = self
    Self.visitChildren { kp, t in t.scale(&out, kp, by: scalar) }
    return out
  }
}

extension Tensor: _VectorProtocol where Scalar: TensorFlowFloatingPoint {}
extension Array.DifferentiableView: _VectorProtocol
where Element: Differentiable & VectorProtocol {}
#endif
