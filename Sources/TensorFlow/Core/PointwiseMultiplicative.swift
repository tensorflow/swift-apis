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

infix operator .*: MultiplicationPrecedence
infix operator .*=: AssignmentPrecedence

/// Implementation detail of the reflection default implementation.
///
/// Contains versions of functions in PointwiseMultiplicative that
/// operate over key paths and modify a child of `Root` in-place.
/// The key paths must all be WritableKeyPath<Root, Self>. This is a workaround
/// to simulate having Self constraints.
public protocol _PointwiseMultiplicative {
  /// lhs[keyPath: kp] .*= rhs[keyPath: kp]
  static func _pointwiseMult<Root>(_ lhs: inout Root, _ rhs: Root, _ kp: PartialKeyPath<Root>)
  /// out[keyPath: kp] = Self.one
  static func _setOne<Root>(_ out: inout Root, _ kp: PartialKeyPath<Root>)
  /// out[keyPath: kp] = out[keyPath: kp].reciprocal
  static func _setReciprocal<Root>(_ out: inout Root, _ kp: PartialKeyPath<Root>)
}

public protocol PointwiseMultiplicative: _PointwiseMultiplicative & AdditiveArithmetic {
  /// The one value.
  ///
  /// One is the identity element for multiplication. For any value,
  /// `x .* .one == x` and `.one .* x == x`.
  static var one: Self { get }

  /// The multiplicative inverse of self.
  ///
  /// For any value, `x .* x.reciprocal == .one` and
  /// `x.reciprocal .* x == .one`.
  var reciprocal: Self { get }

  /// Multiplies two values and produces their product.
  ///
  /// - Parameters:
  ///   - lhs: The first value to multiply.
  ///   - rhs: The second value to multiply.
  static func .* (lhs: Self, rhs: Self) -> Self

  /// Multiplies two values and produces their product.
  ///
  /// - Parameters:
  ///   - lhs: The first value to multiply.
  ///   - rhs: The second value to multiply.
  static func .*= (lhs: inout Self, rhs: Self)
}

extension PointwiseMultiplicative {
  public static func .*= (lhs: inout Self, rhs: Self) {
    lhs = lhs .* rhs
  }
}

extension PointwiseMultiplicative
where Self: ExpressibleByIntegerLiteral {
  public static var one: Self {
    return 1
  }
}

extension PointwiseMultiplicative {
  public static var one: Self {
    var out = self.zero
    visitChildren { kp, t in t._setOne(&out, kp) }
    return out
  }
  public var reciprocal: Self {
    var out = self
    Self.visitChildren { kp, t in t._setReciprocal(&out, kp) }
    return out
  }
  public static func .* (lhs: Self, rhs: Self) -> Self {
    var out = lhs
    visitChildren { kp, t in
      t._pointwiseMult(&out, rhs, kp)
    }
    return out
  }
  public static func _pointwiseMult<Root>(
    _ lhs: inout Root, _ rhs: Root, _ kp: PartialKeyPath<Root>
  ) {
    let kp = kp as! WritableKeyPath<Root, Self>
    lhs[keyPath: kp] .*= rhs[keyPath: kp]
  }
  public static func _setOne<Root>(_ out: inout Root, _ kp: PartialKeyPath<Root>) {
    let kp = kp as! WritableKeyPath<Root, Self>
    out[keyPath: kp] = Self.one
  }
  public static func _setReciprocal<Root>(_ out: inout Root, _ kp: PartialKeyPath<Root>) {
    let kp = kp as! WritableKeyPath<Root, Self>
    out[keyPath: kp] = out[keyPath: kp].reciprocal
  }
}

extension PointwiseMultiplicative {
  internal static func visitChildren(
    _ body: (PartialKeyPath<Self>, _PointwiseMultiplicative.Type) -> Void
  ) {
    guard #available(macOS 9999, *) else {
      fatalError("\(#function) is unavailable")
    }

    if !_forEachFieldWithKeyPath(
      of: Self.self,
      body: { name, kp in
        let valueType = type(of: kp).valueType
        guard let valueType = valueType as? _PointwiseMultiplicative.Type else {
          fatalError("not PointwiseMultiplicative: \(valueType)")
        }
        body(kp, valueType)
        return true
      })
    {
      fatalError(
        "Unreflectable member of \(Self.self) while implementing PointwiseMultiplicative.")
    }
  }
}

extension Array.DifferentiableView: _PointwiseMultiplicative
where Element: Differentiable & PointwiseMultiplicative {}
extension Tensor: _PointwiseMultiplicative where Scalar: Numeric {}

#endif
