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

//===------------------------------------------------------------------------------------------===//
// Method-style differential operators
//===------------------------------------------------------------------------------------------===//

public extension Differentiable {
    @available(*, deprecated, message: """
        Method-style differential operators are deprecated and will be removed; use top-level \
        function 'TensorFlow.gradient(at:in:)' instead
        """)
    @inlinable
    func gradient<R: TensorFlowFloatingPoint>(
        in f: @differentiable (Self) -> Tensor<R>
    ) -> TangentVector {
        return self.valueWithGradient(in: f).1
    }

    @available(*, deprecated, message: """
        Method-style differential operators are deprecated and will be removed; use top-level \
        function 'TensorFlow.valueWithGradient(at:in:)' instead
        """)
    @inlinable
    func valueWithGradient<R: TensorFlowFloatingPoint>(
        in f: @differentiable (Self) -> Tensor<R>
    ) -> (value: Tensor<R>, gradient: TangentVector) {
        let (y, pb) = self.valueWithPullback(in: f)
        precondition(y.rank == 0, """
            The function being differentiated produced a tensor with shape \(y.shape). \
            You can only compute the gradient of functions that return scalar values.
            """)
        return (y, pb(Tensor<R>(1)))
    }

    @available(*, deprecated, message: """
        Method-style differential operators are deprecated and will be removed; use top-level \
        function 'TensorFlow.gradient(at:_:in:)' instead
        """)
    @inlinable
    func gradient<T: Differentiable, R: TensorFlowFloatingPoint>(
        at x: T,
        in f: @differentiable (Self, T) -> Tensor<R>
    ) -> (TangentVector, T.TangentVector) {
        return self.valueWithGradient(at: x, in: f).1
    }

    @available(*, deprecated, message: """
        Method-style differential operators are deprecated and will be removed; use top-level \
        function 'TensorFlow.valueWithGradient(at:_:in:)' instead
        """)
    @inlinable
    func valueWithGradient<T: Differentiable, R: TensorFlowFloatingPoint>(
        at x: T,
        in f: @differentiable (Self, T) -> Tensor<R>
    ) -> (value: Tensor<R>, gradient: (TangentVector, T.TangentVector)) {
        let (y, pb) = self.valueWithPullback(at: x, in: f)
        precondition(y.rank == 0, """
            The function being differentiated produced a tensor with shape \(y.shape). \
            You can only compute the gradient of functions that return scalar values.
            """)
        return (y, pb(Tensor<R>(1)))
    }
}

// ===------------------------------------------------------------------------------------------===//
// Free-function-style differential operators
// ===------------------------------------------------------------------------------------------===//

// Value with gradient

@inlinable
public func valueWithGradient<T, R>(
    at x: T,
    in f: @differentiable (T) -> Tensor<R>
) -> (value: Tensor<R>, gradient: T.TangentVector)
where T: Differentiable, R: TensorFlowFloatingPoint {
    let (y, pullback) = valueWithPullback(at: x, in: f)
    precondition(y.rank == 0, """
        The function being differentiated produced a tensor with shape \(y.shape). \
        You can only compute the gradient of functions that return scalar values.
        """)
    return (y, pullback(Tensor<R>(1)))
}

@inlinable
public func valueWithGradient<T, U, R>(
    at x: T,
    _ y: U,
    in f: @differentiable (T, U) -> Tensor<R>
) -> (value: Tensor<R>, gradient: (T.TangentVector, U.TangentVector))
    where T: Differentiable, U: Differentiable, R: TensorFlowFloatingPoint {
    let (y, pullback) = valueWithPullback(at: x, y, in: f)
    precondition(y.rank == 0, """
        The function being differentiated produced a tensor with shape \(y.shape). \
        You can only compute the gradient of functions that return scalar values.
        """)
    return (y, pullback(Tensor<R>(1)))
}

@inlinable
public func valueWithGradient<T, U, V, R>(
    at x: T,
    _ y: U,
    _ z: V,
    in f: @differentiable (T, U, V) -> Tensor<R>
) -> (value: Tensor<R>, gradient: (T.TangentVector, U.TangentVector, V.TangentVector))
  where T: Differentiable, U: Differentiable, V: Differentiable, R: TensorFlowFloatingPoint {
  let (y, pullback) = valueWithPullback(at: x, y, z, in: f)
  precondition(y.rank == 0)
  return (y, pullback(Tensor<R>(1)))
}

// Value with gradient (curried)

@inlinable
public func valueWithGradient<T, R>(
    of f: @escaping @differentiable (T) -> Tensor<R>
) -> (T) -> (value: Tensor<R>, gradient: T.TangentVector)
    where T: Differentiable, R: TensorFlowFloatingPoint {
    return { x in valueWithGradient(at: x, in: f) }
}

@inlinable
public func valueWithGradient<T, U, R>(
    of f: @escaping @differentiable (T, U) -> Tensor<R>
) -> (T, U) -> (value: Tensor<R>, gradient: (T.TangentVector, U.TangentVector))
  where T: Differentiable, U: Differentiable, R: TensorFlowFloatingPoint {
  return { x, y in valueWithGradient(at: x, y, in: f) }
}

@inlinable
public func valueWithGradient<T, U, V, R>(
    of f: @escaping @differentiable (T, U, V) -> Tensor<R>
) -> (T, U, V) -> (
    value: Tensor<R>,
    gradient: (T.TangentVector, U.TangentVector, V.TangentVector))
    where T: Differentiable, U: Differentiable, V: Differentiable, R: TensorFlowFloatingPoint {
  return { x, y, z in valueWithGradient(at: x, y, z, in: f) }
}

// Gradient

@inlinable
public func gradient<T, R>(
    at x: T,
    in f: @differentiable (T) -> Tensor<R>
) -> T.TangentVector where T: Differentiable, R: TensorFlowFloatingPoint {
    return valueWithGradient(at: x, in: f).1
}

@inlinable
public func gradient<T, U, R>(
    at x: T,
    _ y: U,
    in f: @differentiable (T, U) -> Tensor<R>
) -> (T.TangentVector, U.TangentVector)
    where T: Differentiable, U: Differentiable, R: TensorFlowFloatingPoint {
    return valueWithGradient(at: x, y, in: f).1
}

@inlinable
public func gradient<T, U, V, R>(
    at x: T,
    _ y: U,
    _ z: V,
    in f: @differentiable (T, U, V) -> Tensor<R>
) -> (T.TangentVector, U.TangentVector, V.TangentVector)
    where T: Differentiable, U: Differentiable, V: Differentiable, R: TensorFlowFloatingPoint {
    return valueWithGradient(at: x, y, z, in: f).1
}

// Gradient (curried)

@inlinable
public func gradient<T, R>(
    of f: @escaping @differentiable (T) -> Tensor<R>
) -> (T) -> T.TangentVector where T: Differentiable, R: TensorFlowFloatingPoint {
    return { x in gradient(at: x, in: f) }
}

@inlinable
public func gradient<T, U, R>(
    of f: @escaping @differentiable (T, U) -> Tensor<R>
) -> (T, U) -> (T.TangentVector, U.TangentVector)
    where T: Differentiable, U: Differentiable, R: TensorFlowFloatingPoint {
    return { x, y in gradient(at: x, y, in: f) }
}

@inlinable
public func gradient<T, U, V, R>(
    of f: @escaping @differentiable (T, U, V) -> Tensor<R>
) -> (T, U, V) -> (T.TangentVector, U.TangentVector, V.TangentVector)
    where T: Differentiable, U: Differentiable, V: Differentiable, R: TensorFlowFloatingPoint {
    return { x, y, z in gradient(at: x, y, z, in: f) }
}
