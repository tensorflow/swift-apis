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

#if TENSORFLOW_USE_STANDARD_TOOLCHAIN

import Numerics
@_spi(Reflection) import Swift

extension ElementaryFunctions {
  internal static func visitChildren(
    _ body: (PartialKeyPath<Self>, ElementaryFunctionsVisit.Type) -> Void
  ) {
    guard #available(macOS 9999, *) else {
      fatalError("\(#function) is unavailable")
    }

    if !_forEachFieldWithKeyPath(
      of: Self.self,
      body: { name, kp in
        func visitChild<T>(_: T.Type) {
          guard let t = ElementaryFunctionsVisitor<T>.self as? ElementaryFunctionsVisit.Type
          else {
            fatalError("No conformance of \(T.self) to ElementaryFunctions")
          }
          body(kp, t)
        }
        let valueType = type(of: kp).valueType
        _openExistential(valueType, do: visitChild)
        return true
      })
    {
      fatalError("not all children of \(Self.self) conform to ElementaryFunctions")
    }
  }
}

protocol Functor1 {
  func callAsFunction<T: ElementaryFunctions>(_ x: T) -> T
}
protocol Functor2 {
  func callAsFunction<T: ElementaryFunctions>(_ x: T, _ y: T) -> T
}

protocol ElementaryFunctionsVisit {
  static func applyFunctor<Root, Fn: Functor1>(
    _ out: inout Root, _ kp: PartialKeyPath<Root>, _ fn: Fn)
  static func applyFunctor<Root, Fn: Functor2>(
    _ out: inout Root, _ y: Root, _ kp: PartialKeyPath<Root>, _ fn: Fn)
}
struct ElementaryFunctionsVisitor<T> {}
extension ElementaryFunctionsVisitor: ElementaryFunctionsVisit where T: ElementaryFunctions {
  static func applyFunctor<Root, Fn: Functor1>(
    _ out: inout Root, _ kp: PartialKeyPath<Root>, _ fn: Fn
  ) {
    guard let kp = kp as? WritableKeyPath<Root, T> else { fatalError("problem") }
    ({ (x: inout T) in x = fn(x) })(&out[keyPath: kp])
  }
  static func applyFunctor<Root, Fn: Functor2>(
    _ out: inout Root, _ y: Root, _ kp: PartialKeyPath<Root>, _ fn: Fn
  ) {
    guard let kp = kp as? WritableKeyPath<Root, T> else { fatalError("problem") }
    ({ (x: inout T) in x = fn(x, y[keyPath: kp]) })(&out[keyPath: kp])
  }
}

extension ElementaryFunctions {
  internal init<Fn: Functor1>(mapped fn: Fn, _ x: Self) {
    self = x
    Self.visitChildren { kp, t in t.applyFunctor(&self, kp, fn) }
  }
  internal init<Fn: Functor2>(mapped fn: Fn, _ x: Self, _ y: Self) {
    self = x
    Self.visitChildren { kp, t in t.applyFunctor(&self, y, kp, fn) }
  }
}

struct Functor_exp: Functor1 {
  func callAsFunction<T: ElementaryFunctions>(_ x: T) -> T { T.exp(x) }
}
struct Functor_expMinusOne: Functor1 {
  func callAsFunction<T: ElementaryFunctions>(_ x: T) -> T { T.expMinusOne(x) }
}
struct Functor_cosh: Functor1 {
  func callAsFunction<T: ElementaryFunctions>(_ x: T) -> T { T.cosh(x) }
}
struct Functor_sinh: Functor1 {
  func callAsFunction<T: ElementaryFunctions>(_ x: T) -> T { T.sinh(x) }
}
struct Functor_tanh: Functor1 {
  func callAsFunction<T: ElementaryFunctions>(_ x: T) -> T { T.tanh(x) }
}
struct Functor_cos: Functor1 {
  func callAsFunction<T: ElementaryFunctions>(_ x: T) -> T { T.cos(x) }
}
struct Functor_sin: Functor1 {
  func callAsFunction<T: ElementaryFunctions>(_ x: T) -> T { T.sin(x) }
}
struct Functor_tan: Functor1 {
  func callAsFunction<T: ElementaryFunctions>(_ x: T) -> T { T.tan(x) }
}
struct Functor_log: Functor1 {
  func callAsFunction<T: ElementaryFunctions>(_ x: T) -> T { T.log(x) }
}
struct Functor_log1p: Functor1 {
  func callAsFunction<T: ElementaryFunctions>(_ x: T) -> T { T.log(onePlus: x) }
}
struct Functor_acosh: Functor1 {
  func callAsFunction<T: ElementaryFunctions>(_ x: T) -> T { T.acosh(x) }
}
struct Functor_asinh: Functor1 {
  func callAsFunction<T: ElementaryFunctions>(_ x: T) -> T { T.asinh(x) }
}
struct Functor_atanh: Functor1 {
  func callAsFunction<T: ElementaryFunctions>(_ x: T) -> T { T.atanh(x) }
}
struct Functor_acos: Functor1 {
  func callAsFunction<T: ElementaryFunctions>(_ x: T) -> T { T.acos(x) }
}
struct Functor_asin: Functor1 {
  func callAsFunction<T: ElementaryFunctions>(_ x: T) -> T { T.asin(x) }
}
struct Functor_atan: Functor1 {
  func callAsFunction<T: ElementaryFunctions>(_ x: T) -> T { T.atan(x) }
}
struct Functor_sqrt: Functor1 {
  func callAsFunction<T: ElementaryFunctions>(_ x: T) -> T { T.sqrt(x) }
}
struct Functor_pow: Functor1 {
  var n: Int
  func callAsFunction<T: ElementaryFunctions>(_ x: T) -> T { T.pow(x, n) }
}
struct Functor_pow2: Functor2 {
  func callAsFunction<T: ElementaryFunctions>(_ x: T, _ y: T) -> T { T.pow(x, y) }
}
struct Functor_root: Functor1 {
  var n: Int
  func callAsFunction<T: ElementaryFunctions>(_ x: T) -> T { T.root(x, n) }
}

extension ElementaryFunctions {
  public static func exp(_ x: Self) -> Self { .init(mapped: Functor_exp(), x) }
  public static func expMinusOne(_ x: Self) -> Self { .init(mapped: Functor_expMinusOne(), x) }
  public static func tanh(_ x: Self) -> Self { .init(mapped: Functor_tanh(), x) }
  public static func cosh(_ x: Self) -> Self { .init(mapped: Functor_cosh(), x) }
  public static func sinh(_ x: Self) -> Self { .init(mapped: Functor_sinh(), x) }
  public static func cos(_ x: Self) -> Self { .init(mapped: Functor_cos(), x) }
  public static func sin(_ x: Self) -> Self { .init(mapped: Functor_sin(), x) }
  public static func tan(_ x: Self) -> Self { .init(mapped: Functor_tan(), x) }
  public static func log(_ x: Self) -> Self { .init(mapped: Functor_log(), x) }
  public static func log(onePlus x: Self) -> Self { .init(mapped: Functor_log1p(), x) }
  public static func acosh(_ x: Self) -> Self { .init(mapped: Functor_acosh(), x) }
  public static func asinh(_ x: Self) -> Self { .init(mapped: Functor_asinh(), x) }
  public static func atanh(_ x: Self) -> Self { .init(mapped: Functor_atanh(), x) }
  public static func acos(_ x: Self) -> Self { .init(mapped: Functor_acos(), x) }
  public static func asin(_ x: Self) -> Self { .init(mapped: Functor_asin(), x) }
  public static func atan(_ x: Self) -> Self { .init(mapped: Functor_atan(), x) }
  public static func sqrt(_ x: Self) -> Self { .init(mapped: Functor_sqrt(), x) }
  public static func pow(_ x: Self, _ n: Int) -> Self { .init(mapped: Functor_pow(n: n), x) }
  public static func root(_ x: Self, _ n: Int) -> Self { .init(mapped: Functor_root(n: n), x) }
  public static func pow(_ x: Self, _ y: Self) -> Self { .init(mapped: Functor_pow2(), x, y) }
}

#endif
