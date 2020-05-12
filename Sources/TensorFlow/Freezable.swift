// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

/// A wrapper around a value whose value can be frozen.
///
/// When `isFrozen` is true, assignments to `wrappedValue` will do nothing.
@propertyWrapper
public struct _Freezable<Value> {
  private var _value: Value

  /// True iff the value is frozen.
  public var isFrozen: Bool = false

  public init(wrappedValue: Value) {
    _value = wrappedValue
  }

  public var projectedValue: Self {
    get { self }
    set { self = newValue }
  }

  /// The wrapped value.
  public var wrappedValue: Value {
    get { _value }
    set {
      // If frozen, do not update the value.
      if isFrozen { return }
      // Otherwise, update the value.
      _value = newValue
    }
  }
}

extension _Freezable {
  /// Freeze the value of `wrappedValue`.
  ///
  /// While frozen, assignments to `wrappedValue` will do nothing.
  public mutating func freeze() {
    isFrozen = true
  }

  /// Unfreeze the value of `wrappedValue`.
  ///
  /// Assignments to `wrappedValue` will behave as normal.
  public mutating func unfreeze() {
    isFrozen = false
  }
}
