// Copyright 2020 TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// A floating-point data type that conforms to `Differentiable` and is compatible with TensorFlow.
///
/// - Note: `Tensor` conditionally conforms to `Differentiable` when the `Scalar` associated type
///   conforms `TensorFlowFloatingPoint`.
public protocol TensorFlowFloatingPoint:
  XLAScalarType & BinaryFloatingPoint & Differentiable & ElementaryFunctions
where
  Self.RawSignificand: FixedWidthInteger,
  Self == Self.TangentVector
{}
public typealias TensorFlowNumeric = XLAScalarType & Numeric
public typealias TensorFlowScalar = XLAScalarType
public typealias TensorFlowInteger = TensorFlowScalar & BinaryInteger

/// An integer data type that represents integer types which can be used as tensor indices in 
/// TensorFlow.
public protocol TensorFlowIndex: TensorFlowInteger {}

extension Int32: TensorFlowIndex {}
extension Int64: TensorFlowIndex {}

extension Float: TensorFlowFloatingPoint {}
extension Double: TensorFlowFloatingPoint {}
