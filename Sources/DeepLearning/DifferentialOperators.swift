// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

import TensorFlowCore

//===------------------------------------------------------------------------------------------===//
// Method-style Differential Operators
//===------------------------------------------------------------------------------------------===//

public extension Differentiable {
    @inlinable
    func gradient<R: TensorFlowFloatingPoint>(
        in f: @differentiable (Self) -> Tensor<R>
    ) -> CotangentVector {
        return self.pullback(in: f)(Tensor<R>(1))
    }

    @inlinable
    func valueWithGradient<R: TensorFlowFloatingPoint>(
        in f: @differentiable (Self) -> Tensor<R>
    ) -> (value: Tensor<R>, gradient: CotangentVector) {
        let (y, pb) = self.valueWithPullback(in: f)
        return (y, pb(Tensor<R>(1)))
    }

    @inlinable
    func gradient<T: Differentiable, R: TensorFlowFloatingPoint>(
        at x: T,
        in f: @differentiable (Self, T) -> Tensor<R>
    ) -> (CotangentVector, T.CotangentVector) {
        return self.pullback(at: x, in: f)(Tensor<R>(1))
    }

    @inlinable
    func valueWithGradient<T: Differentiable, R: TensorFlowFloatingPoint>(
        at x: T,
        in f: @differentiable (Self, T) -> Tensor<R>
    ) -> (value: Tensor<R>, gradient: (CotangentVector, T.CotangentVector)) {
        let (y, pb) = self.valueWithPullback(at: x, in: f)
        return (y, pb(Tensor<R>(1)))
    }
}

//===------------------------------------------------------------------------------------------===//
// Free-Function-Style Differential Operators
//===------------------------------------------------------------------------------------------===//

// Value with gradient

@inlinable
public func valueWithGradient<T, R>(
    at x: T,
    in f: @differentiable (T) -> Tensor<R>
) -> (value: Tensor<R>, gradient: T.CotangentVector)
where T: Differentiable, R: TensorFlowFloatingPoint {
    let (y, pullback) = valueWithPullback(at: x, in: f)
    return (y, pullback(Tensor<R>(1)))
}

@inlinable
public func valueWithGradient<T, U, R>(
    at x: T,
    _ y: U,
    in f: @differentiable (T, U) -> Tensor<R>
) -> (value: Tensor<R>, gradient: (T.CotangentVector, U.CotangentVector))
    where T: Differentiable, U: Differentiable, R: TensorFlowFloatingPoint {
    let (y, pullback) = valueWithPullback(at: x, y, in: f)
    return (y, pullback(Tensor<R>(1)))
}

@inlinable
public func valueWithGradient<T, U, V, R>(
    at x: T,
    _ y: U,
    _ z: V,
    in f: @differentiable (T, U, V) -> Tensor<R>
) -> (value: Tensor<R>, gradient: (T.CotangentVector, U.CotangentVector, V.CotangentVector))
  where T: Differentiable, U: Differentiable, V: Differentiable, R: TensorFlowFloatingPoint {
  let (y, pullback) = valueWithPullback(at: x, y, z, in: f)
  return (y, pullback(Tensor<R>(1)))
}

// Value with gradient (curried)

@inlinable
public func valueWithGradient<T, R>(
    of f: @escaping @differentiable (T) -> Tensor<R>
) -> (T) -> (value: Tensor<R>, gradient: T.CotangentVector)
    where T: Differentiable, R: TensorFlowFloatingPoint {
    return { x in valueWithGradient(at: x, in: f) }
}

@inlinable
public func valueWithGradient<T, U, R>(
    of f: @escaping @differentiable (T, U) -> Tensor<R>
) -> (T, U) -> (value: Tensor<R>, gradient: (T.CotangentVector, U.CotangentVector))
  where T: Differentiable, U: Differentiable, R: TensorFlowFloatingPoint {
  return { x, y in valueWithGradient(at: x, y, in: f) }
}

@inlinable
public func valueWithGradient<T, U, V, R>(
    of f: @escaping @differentiable (T, U, V) -> Tensor<R>
) -> (T, U, V) -> (
    value: Tensor<R>,
    gradient: (T.CotangentVector, U.CotangentVector, V.CotangentVector))
    where T: Differentiable, U: Differentiable, V: Differentiable, R: TensorFlowFloatingPoint {
  return { x, y, z in valueWithGradient(at: x, y, z, in: f) }
}

// Gradient

@inlinable
public func gradient<T, R>(
    at x: T,
    in f: @differentiable (T) -> Tensor<R>
) -> T.CotangentVector where T: Differentiable, R: TensorFlowFloatingPoint {
    return pullback(at: x, in: f)(Tensor<R>(1))
}

@inlinable
public func gradient<T, U, R>(
    at x: T,
    _ y: U,
    in f: @differentiable (T, U) -> Tensor<R>
) -> (T.CotangentVector, U.CotangentVector)
    where T: Differentiable, U: Differentiable, R: TensorFlowFloatingPoint {
    return pullback(at: x, y, in: f)(Tensor<R>(1))
}

@inlinable
public func gradient<T, U, V, R>(
    at x: T,
    _ y: U,
    _ z: V,
    in f: @differentiable (T, U, V) -> Tensor<R>
) -> (T.CotangentVector, U.CotangentVector, V.CotangentVector)
    where T: Differentiable, U: Differentiable, V: Differentiable, R: TensorFlowFloatingPoint {
    return pullback(at: x, y, z, in: f)(Tensor<R>(1))
}

// Gradient (curried)

@inlinable
public func gradient<T, R>(
    of f: @escaping @differentiable (T) -> Tensor<R>
) -> (T) -> T.CotangentVector where T: Differentiable, R: TensorFlowFloatingPoint {
    return { x in gradient(at: x, in: f) }
}

@inlinable
public func gradient<T, U, R>(
    of f: @escaping @differentiable (T, U) -> Tensor<R>
) -> (T, U) -> (T.CotangentVector, U.CotangentVector)
    where T: Differentiable, U: Differentiable, R: TensorFlowFloatingPoint {
    return { x, y in gradient(at: x, y, in: f) }
}

@inlinable
public func gradient<T, U, V, R>(
    of f: @escaping @differentiable (T, U, V) -> Tensor<R>
) -> (T, U, V) -> (T.CotangentVector, U.CotangentVector, V.CotangentVector)
    where T: Differentiable, U: Differentiable, V: Differentiable, R: TensorFlowFloatingPoint {
    return { x, y, z in gradient(at: x, y, z, in: f) }
}
