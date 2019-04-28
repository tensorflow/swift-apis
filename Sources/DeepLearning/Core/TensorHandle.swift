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

internal extension TensorHandle {
    /// Create a `ShapedArray` with contents of the underlying `TensorHandle`. If
    /// the `TensorHandle` is on the accelerator, it will be copied to the host.
    /// - Returns: A `ShapedArray`.
    @usableFromInline
    @inline(never)
    func makeHostCopy() -> ShapedArray<Scalar> {
        internalConsistencyCheck(isConcrete)
        debugLog("Calling makeHostCopy() with c handle \(_cTensorHandle)")
        return ShapedArray(cTensorHandle: _cTensorHandle)
    }
}
