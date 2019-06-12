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
    /// Computes `lhs < rhs` element-wise and returns a `Tensor` of Boolean /// scalars.
    @inlinable
    static func .< (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
        return Raw.less(lhs, rhs)
    }

    /// Computes `lhs <= rhs` element-wise and returns a `Tensor` of Boolean scalars.
    @inlinable
    static func .<= (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
        return Raw.lessEqual(lhs, rhs)
    }

    /// Computes `lhs > rhs` element-wise and returns a `Tensor` of Boolean scalars.
    @inlinable
    static func .> (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
        return Raw.greater(lhs, rhs)
    }

    /// Computes `lhs >= rhs` element-wise and returns a `Tensor` of Boolean scalars.
    @inlinable
    static func .>= (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
        return Raw.greaterEqual(lhs, rhs)
    }

    /// Computes `lhs < rhs` element-wise and returns a `Tensor` of Boolean scalars.
    /// - Note: `.<` supports broadcasting.
    @inlinable
    static func .< (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
        return Raw.less(Tensor(lhs), rhs)
    }

    /// Computes `lhs <= rhs` element-wise and returns a `Tensor` of Boolean scalars.
    /// - Note: `.<=` supports broadcasting.
    @inlinable
    static func .<= (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
        return Raw.lessEqual(Tensor(lhs), rhs)
    }

    /// Computes `lhs > rhs` element-wise and returns a `Tensor` of Boolean scalars.
    /// - Note: `.>` supports broadcasting.
    @inlinable
    static func .> (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
        return Raw.greater(Tensor(lhs), rhs)
    }

    /// Computes `lhs >= rhs` element-wise and returns a `Tensor` of Boolean scalars.
    /// - Note: `.>=` supports broadcasting.
    @inlinable
    static func .>= (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
        return Raw.greaterEqual(Tensor(lhs), rhs)
    }

    /// Computes `lhs < rhs` element-wise and returns a `Tensor` of Boolean scalars.
    /// - Note: `.<` supports broadcasting.
    @inlinable
    static func .< (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
        return Raw.less(lhs, Tensor(rhs))
    }

    /// Computes `lhs <= rhs` element-wise and returns a `Tensor` of Boolean scalars.
    /// - Note: `.<=` supports broadcasting.
    @inlinable
    static func .<= (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
        return Raw.lessEqual(lhs, Tensor(rhs))
    }

    /// Computes `lhs > rhs` element-wise and returns a `Tensor` of Boolean scalars.
    /// - Note: `.>` supports broadcasting.
    @inlinable
    static func .> (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
        return Raw.greater(lhs, Tensor(rhs))
    }

    /// Computes `lhs >= rhs` element-wise and returns a `Tensor` of Boolean scalars.
    /// - Note: `.>=` supports broadcasting.
    @inlinable
    static func .>= (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
        return Raw.greaterEqual(lhs, Tensor(rhs))
    }
}

extension Tensor: Comparable where Scalar: Numeric & Comparable {
    /// Returns a Boolean value indicating whether the value of the first argument is
    /// lexicographically less than that of the second argument.
    @inlinable
    public static func < (lhs: Tensor, rhs: Tensor) -> Bool {
        return (lhs .< rhs).all()
    }

    /// Returns a Boolean value indicating whether the value of the first argument is
    /// lexicographically less than or equal to that of the second argument.
    @inlinable
    public static func <= (lhs: Tensor, rhs: Tensor) -> Bool {
        return (lhs .<= rhs).all()
    }

    /// Returns a Boolean value indicating whether the value of the first argument is
    /// lexicographically greater than that of the second argument.
    @inlinable
    public static func > (lhs: Tensor, rhs: Tensor) -> Bool {
        return (lhs .> rhs).all()
    }

    /// Returns a Boolean value indicating whether the value of the first argument is
    /// lexicographically greater than or equal to that of the second argument.
    @inlinable
    public static func >= (lhs: Tensor, rhs: Tensor) -> Bool {
        return (lhs .>= rhs).all()
    }
}

public extension Tensor where Scalar: Numeric & Comparable {
    /// Returns a Boolean value indicating whether the value of the first argument is
    /// lexicographically less than that of the second argument.
    @inlinable
    static func < (lhs: Tensor, rhs: Scalar) -> Bool {
        return (lhs .< rhs).all()
    }

    /// Returns a Boolean value indicating whether the value of the first argument is
    /// lexicographically less than or equal to that of the second argument.
    @inlinable
    static func <= (lhs: Tensor, rhs: Scalar) -> Bool {
        return (lhs .<= rhs).all()
    }

    /// Returns a Boolean value indicating whether the value of the first argument is
    /// lexicographically greater than that of the second argument.
    @inlinable
    static func > (lhs: Tensor, rhs: Scalar) -> Bool {
        return (lhs .> rhs).all()
    }

    /// Returns a Boolean value indicating whether the value of the first argument is
    /// lexicographically greater than or equal to that of the second argument.
    @inlinable
    static func >= (lhs: Tensor, rhs: Scalar) -> Bool {
        return (lhs .>= rhs).all()
    }
}

public extension Tensor where Scalar: Equatable {
    /// Computes `lhs != rhs` element-wise and returns a `Tensor` of Boolean scalars.
    /// - Note: `.==` supports broadcasting.
    @inlinable
    static func .== (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
        return Raw.equal(lhs, rhs)
    }

    /// Computes `lhs != rhs` element-wise and returns a `Tensor` of Boolean scalars.
    /// - Note: `.!=` supports broadcasting.
    @inlinable
    static func .!= (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
        return Raw.notEqual(lhs, rhs)
    }

    /// Computes `lhs == rhs` element-wise and returns a `Tensor` of Boolean scalars.
    /// - Note: `.==` supports broadcasting.
    @inlinable
    static func .== (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
        return Tensor(lhs) .== rhs
    }

    /// Computes `lhs != rhs` element-wise and returns a `Tensor` of Boolean scalars.
    /// - Note: `.!=` supports broadcasting.
    @inlinable
    static func .!= (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
        return Tensor(lhs) .!= rhs
    }

    /// Computes `lhs == rhs` element-wise and returns a `Tensor` of Boolean
    /// scalars.
    /// - Note: `.==` supports broadcasting.
    @inlinable
    static func .== (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
        return lhs .== Tensor(rhs)
    }

    /// Computes `lhs != rhs` element-wise and returns a `Tensor` of Boolean scalars.
    /// - Note: `.!=` supports broadcasting.
    @inlinable
    static func .!= (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
        return lhs .!= Tensor(rhs)
    }
}

// TODO: infix operator ≈: ComparisonPrecedence

public extension Tensor where Scalar: FloatingPoint & Equatable {
    /// Returns a `Tensor` of Boolean values indicating whether the elements of `self` are
    /// approximately equal to those of `other`.
    @inlinable
    func elementsApproximatelyEqual(
        _ other: Tensor,
        tolerance: Double = 0.00001
    ) -> Tensor<Bool> {
        return Raw.approximateEqual(self, other, tolerance: tolerance)
    }
}

public extension StringTensor {
    /// Computes `self == other` element-wise.
    /// - Note: `elementsEqual` supports broadcasting.
    @inlinable
    func elementsEqual(_ other: StringTensor) -> Tensor<Bool> {
        return Raw.equal(self, other)
    }
}
