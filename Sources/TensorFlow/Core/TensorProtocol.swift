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

public protocol TensorProtocol {
    /// Scalar type.
    associatedtype Scalar: TensorFlowScalar

    /// The underlying `TensorHandle`.
    /// - Note: Do NOT remove this. This is a compiler requirement.
    var handle: TensorHandle<Scalar> { get }

    /// Initialize from a `TensorHandle`.
    /// - Note: Do NOT remove this. This is a compiler requirement.
    init(handle: TensorHandle<Scalar>)
}

/// The protocol on tensor sends and receives.
///
/// - Note: The compiler knows about this protocol and generates code to use it. So changing the 
///   protocol design requires changing the compiler accordingly too.
protocol TensorSendableReceivable {
    associatedtype Scalar

    /// Receive a tensor based on a tensor computation handle (equivalent to a TF session handle),
    /// and a tensor ID.
    static func receiveFromAccelerator(_ computation: _TensorComputation, _ tensorID: Int) -> Self

    /// Send a tensor of `this` instance based on a tensor computation handle (equivalent to a TF
    /// session handle), and a tensor ID.
    func sendToAccelerator(_ computation: _TensorComputation, _ tensorID: Int)

    /// Create a scalar tensor. It can be used by the host program to send a scalar value to
    /// accelerator.
    ///
    /// - Note: This is different from protocol method `TensorFlowScalar._makeScalarTensor()`, a 
    ///   marker function to assist compiler in generating Accelerator code, and has no runtime
    ///   effects.
    static func scalar(_ scalar: Scalar) -> Self
}

