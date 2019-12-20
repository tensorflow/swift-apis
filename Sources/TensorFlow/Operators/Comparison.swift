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

infix operator .<: ComparisonPrecedence
infix operator .<=: ComparisonPrecedence
infix operator .>=: ComparisonPrecedence
infix operator .>: ComparisonPrecedence
infix operator .==: ComparisonPrecedence
infix operator .!=: ComparisonPrecedence

public extension Tensor where Scalar: Numeric & Comparable {
    /// Returns a tensor of Boolean scalars by computing `lhs < rhs` element-wise.
    @inlinable
    static func .< (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
        return _Raw.less(lhs, rhs)
    }

    /// Returns a tensor of Boolean scalars by computing `lhs <= rhs` element-wise.
    @inlinable
    static func .<= (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
        return _Raw.lessEqual(lhs, rhs)
    }

    /// Returns a tensor of Boolean scalars by computing `lhs > rhs` element-wise.
    @inlinable
    static func .> (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
        return _Raw.greater(lhs, rhs)
    }

    /// Returns a tensor of Boolean scalars by computing `lhs >= rhs` element-wise.
    @inlinable
    static func .>= (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
        return _Raw.greaterEqual(lhs, rhs)
    }

    /// Returns a tensor of Boolean scalars by computing `lhs < rhs` element-wise.
    /// - Note: `.<` supports broadcasting.
    @inlinable
    static func .< (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
        return _Raw.less(Tensor(lhs, on: rhs.device), rhs)
    }

    /// Returns a tensor of Boolean scalars by computing `lhs <= rhs` element-wise.
    /// - Note: `.<=` supports broadcasting.
    @inlinable
    static func .<= (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
        return _Raw.lessEqual(Tensor(lhs, on: rhs.device), rhs)
    }

    /// Returns a tensor of Boolean scalars by computing `lhs > rhs` element-wise.
    /// - Note: `.>` supports broadcasting.
    @inlinable
    static func .> (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
        return _Raw.greater(Tensor(lhs, on: rhs.device), rhs)
    }

    /// Returns a tensor of Boolean scalars by computing `lhs >= rhs` element-wise.
    /// - Note: `.>=` supports broadcasting.
    @inlinable
    static func .>= (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
        return _Raw.greaterEqual(Tensor(lhs, on: rhs.device), rhs)
    }

    /// Returns a tensor of Boolean scalars by computing `lhs < rhs` element-wise.
    /// - Note: `.<` supports broadcasting.
    @inlinable
    static func .< (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
        return _Raw.less(lhs, Tensor(rhs, on: lhs.device))
    }

    /// Returns a tensor of Boolean scalars by computing `lhs <= rhs` element-wise.
    /// - Note: `.<=` supports broadcasting.
    @inlinable
    static func .<= (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
        return _Raw.lessEqual(lhs, Tensor(rhs, on: lhs.device))
    }

    /// Returns a tensor of Boolean scalars by computing `lhs > rhs` element-wise.
    /// - Note: `.>` supports broadcasting.
    @inlinable
    static func .> (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
        return _Raw.greater(lhs, Tensor(rhs, on: lhs.device))
    }

    /// Returns a tensor of Boolean scalars by computing `lhs >= rhs` element-wise.
    /// - Note: `.>=` supports broadcasting.
    @inlinable
    static func .>= (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
        return _Raw.greaterEqual(lhs, Tensor(rhs, on: lhs.device))
    }
}

public extension Tensor where Scalar: Equatable {
    /// Returns a tensor of Boolean scalars by computing `lhs == rhs` element-wise.
    /// - Note: `.==` supports broadcasting.
    @inlinable
    static func .== (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
        return _Raw.equal(lhs, rhs)
    }

    /// Returns a tensor of Boolean scalars by computing `lhs != rhs` element-wise.
    /// - Note: `.!=` supports broadcasting.
    @inlinable
    static func .!= (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
        return _Raw.notEqual(lhs, rhs)
    }

    /// Returns a tensor of Boolean scalars by computing `lhs == rhs` element-wise.
    /// - Note: `.==` supports broadcasting.
    @inlinable
    static func .== (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
        return Tensor(lhs, on: rhs.device) .== rhs
    }

    /// Returns a tensor of Boolean scalars by computing `lhs != rhs` element-wise.
    /// - Note: `.!=` supports broadcasting.
    @inlinable
    static func .!= (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
        return Tensor(lhs, on: rhs.device) .!= rhs
    }

    /// Returns a tensor of Boolean scalars by computing `lhs == rhs` element-wise.
    /// - Note: `.==` supports broadcasting.
    @inlinable
    static func .== (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
        return lhs .== Tensor(rhs, on: lhs.device)
    }

    /// Returns a tensor of Boolean scalars by computing `lhs != rhs` element-wise.
    /// - Note: `.!=` supports broadcasting.
    @inlinable
    static func .!= (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
        return lhs .!= Tensor(rhs, on: lhs.device)
    }
}

// TODO: infix operator ≈: ComparisonPrecedence

public extension Tensor where Scalar: TensorFlowFloatingPoint & Equatable {
    /// Returns a tensor of Boolean values indicating whether the elements of `self` are
    /// approximately equal to those of `other`.
    /// - Precondition: `self` and `other` must be of the same shape.
    @inlinable
    func elementsAlmostEqual(
        _ other: Tensor,
        tolerance: Scalar = Scalar.ulpOfOne.squareRoot()
    ) -> Tensor<Bool> {
        return _Raw.approximateEqual(self, other, tolerance: Double(tolerance))
    }
}

public extension Tensor where Scalar: TensorFlowFloatingPoint {
    /// Returns `true` if all elements of `self` are approximately equal to those of `other`.
    /// - Precondition: `self` and `other` must be of the same shape.
    @inlinable
    func isAlmostEqual(
        to other: Tensor,
        tolerance: Scalar = Scalar.ulpOfOne.squareRoot()
    ) -> Bool {
        elementsAlmostEqual(other, tolerance: tolerance).all()
    }
}
