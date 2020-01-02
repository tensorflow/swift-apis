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

/// A wrapper around a differentiable value with "freezable" derivatives.
///
/// When `isFrozen` is true, accesses to `wrappedValue` have a derivative of zero.
@propertyWrapper
public struct _Freezable<Value: Differentiable> {
    @noDerivative public var isFrozen: Bool = false
    private var _value: Value

    public init(wrappedValue: Value) {
        _value = wrappedValue
    }

    public var projectedValue: Self {
        get { return self }
        set { self = newValue }
    }

    /// The wrapped differentiable value.
    @differentiable
    public var wrappedValue: Value {
        get { _value }
        set { _value = newValue }
    }

    @usableFromInline
    @derivative(of: wrappedValue)
    func _vjpValue() -> (value: Value, pullback: (Value.TangentVector) -> TangentVector) {
        return (_value, { [isFrozen = self.isFrozen] v in
            isFrozen ? .zero : v
        })
    }
}

extension _Freezable {
    /// Freeze derivatives for `wrappedValue`. Accesses to `wrappedValue` will always have a
    /// derivative of zero.
    public mutating func freeze() {
        isFrozen = true
    }

    /// Unfreeze derivatives for `wrappedValue`.
    public mutating func unfreeze() {
        isFrozen = false
    }
}

extension _Freezable: Differentiable {
    public typealias TangentVector = Value.TangentVector
    public mutating func move(along direction: TangentVector) {
        _value.move(along: direction)
    }
}

extension _Freezable: EuclideanDifferentiable where Value: EuclideanDifferentiable {
    public var differentiableVectorView: TangentVector {
         return _value.differentiableVectorView
    }
}
