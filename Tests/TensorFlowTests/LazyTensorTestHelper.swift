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

@testable import TensorFlow

protocol _LazyTensorCompatible {
    /// The underlying `LazyTensorHandle` (if any).
    var _lazyTensor: LazyTensorHandle? { get }

    /// Returns `Self` that wraps a concrete `LazyTensorHandle`.
    /// (Triggers materialization if needed.)
    var _concreteLazyTensor: Self { get }

    /// Similar to the `concreteLazyTensor` with an additional constraint that
    /// the underlying concrete `LazyTensorHandle` should be marked to be promoted as
    /// an input when used in an extracted trace.
    var _concreteInputLazyTensor: Self { get }
}

extension _AnyTensorHandle {
    var _lazyTensor: LazyTensorHandle? {
        if let handle = self as? LazyTensorHandle {
            return handle
        } else {
            return nil
        }
    }
    var _concreteLazyTensor: LazyTensorHandle { LazyTensorHandle(self._tfeTensorHandle) }
}

extension TensorHandle: _LazyTensorCompatible {
    var _lazyTensor: LazyTensorHandle? { handle._lazyTensor }
    public var _concreteLazyTensor: TensorHandle {
        TensorHandle(handle: handle._concreteLazyTensor)
    }
}

extension Tensor: _LazyTensorCompatible {
    var _lazyTensor: LazyTensorHandle? { handle._lazyTensor }
    public var _concreteLazyTensor: Tensor {
        Tensor(handle: handle._concreteLazyTensor)
    }
}

extension StringTensor: _LazyTensorCompatible {
    var _lazyTensor: LazyTensorHandle? { handle._lazyTensor }
    public var _concreteLazyTensor: StringTensor {
        StringTensor(handle: handle._concreteLazyTensor)
    }
}

extension VariantHandle: _LazyTensorCompatible {
    var _lazyTensor: LazyTensorHandle? { handle._lazyTensor }
    public var _concreteLazyTensor: VariantHandle {
        VariantHandle(handle: handle._concreteLazyTensor)
    }
}

extension ResourceHandle: _LazyTensorCompatible {
    var _lazyTensor: LazyTensorHandle? { handle._lazyTensor }
    public var _concreteLazyTensor: ResourceHandle {
        ResourceHandle(handle: handle._concreteLazyTensor)
    }
}
