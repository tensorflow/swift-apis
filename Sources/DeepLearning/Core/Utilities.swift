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

#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
import Darwin
#else
import Glibc
#endif

@_exported import TensorFlowCore

//===----------------------------------------------------------------------===//
// Unit test utilities
//===----------------------------------------------------------------------===//
// TODO: Move this section to a unit-test only Swift module, once the google
// internal lit based test infra can handle importing additional Swift modules.

/// This is a generic host-only op that hides the details of its impl in the SIL
/// code. This makes reading/writing SIL based compiler unit tests simple.
@inline(never)
public func _hostOp<T>(_ x: T) {
  print(x)
}

@inline(never)
public func _hostOp<Scalar>(_ x: Tensor<Scalar>) {
  print(x)
}

@inline(never)
public func _hostOp<Scalar : TensorFlowScalar>(_ x: TensorHandle<Scalar>) {
  print(Tensor(handle: x))
}

/// Some TPU ops (e.g. infeed/outfeed) require tensor shape info, which the APIs
/// below can provide.
///
/// TODO: Remove these helper APIs, when we have a better shape
/// inference/propagation design.
@inlinable @inline(__always)
public func _scalarTensorWithShape<Scalar>(
  _ x: Tensor<Scalar>
) -> Tensor<Scalar> {
  let ret: TensorHandle<Scalar> =
    #tfop("Identity", x, T$dtype: Scalar.tensorFlowDataType,
          __shapes: [TensorShape()])
  return Tensor<Scalar>(handle: ret)
}

@inlinable @inline(__always)
public func _addScalarTensorsWithShape<Scalar>(
  _ x: Tensor<Scalar>,
  _ y: Tensor<Scalar>
) -> Tensor<Scalar> {
  let ret: TensorHandle<Scalar> =
    #tfop("Add", x, y, T$dtype: Scalar.tensorFlowDataType,
          __shapes: [TensorShape()])
  return Tensor<Scalar>(handle: ret)
}
