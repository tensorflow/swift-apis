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

#if !COMPILING_TENSORFLOW_MODULE
import TensorFlow
#endif

//===------------------------------------------------------------------------------------------===//
// Shape Transformations
//===------------------------------------------------------------------------------------------===//

public extension Tensor {
    /// Convert to a tensor with the specified rank, with all dimensions equal to 1.
    @inlinable
    func makeTensor(rank: Int) -> Tensor<Self> {
        return Tensor(repeating: self, shape: TensorShape(rank))
    }

    /// Reshape to the shape of the specified `Tensor`.
    /// - Precondition: The number of scalars matches the new shape.
    @inlinable
    @differentiable(wrt: self where Scalar : TensorFlowFloatingPoint)
    func reshaped<T>(like other: Tensor<T>) -> Tensor {
        return reshaped(toShape: other.shapeTensor)
    }

    /// Reshape to the specified shape.
    /// - Precondition: The number of scalars matches the new shape.
    @inlinable
    @differentiable(wrt: self where Scalar : TensorFlowFloatingPoint)
    func reshaped(to newShape: TensorShape) -> Tensor {
        // TODO(TF-433): Remove workaround for differentiating `map`.
        return reshaped(toShape: Tensor<Int32>({newShape.dimensions.map(Int32.init)}()))
    }

    /// Reshape to the specified `Tensor` representing a shape.
    /// - Precondition: The number of scalars matches the new shape.
    @inlinable
    @differentiable(
        wrt: self,
        vjp: _vjpReshaped(toShape:) where Scalar : TensorFlowFloatingPoint)
    func reshaped(toShape newShape: Tensor<Int32>) -> Tensor {
        return Raw.reshape(self, shape: newShape)
    }

    /// Return a copy of the tensor collapsed into a 1-D `Tensor`, in row-major order.
    @inlinable
    @differentiable(wrt: self where Scalar : TensorFlowFloatingPoint)
    func flattened() -> Tensor {
        return reshaped(to: [-1])
    }

    /// Returns a shape-expanded `Tensor`, with a dimension of 1 inserted at the
    /// specified shape index.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpExpandingShape(at:) where Scalar : TensorFlowFloatingPoint)
    func expandingShape(at shapeIndex: Int) -> Tensor {
        return Raw.expandDims(self, dim: Tensor<Int32>(Int32(shapeIndex)))
    }

    /// Returns a rank-lifted `Tensor` with a leading dimension of 1.
    @inlinable
    @differentiable(wrt: self where Scalar : TensorFlowFloatingPoint)
    func rankLifted() -> Tensor {
        return expandingShape(at: 0)
    }

    /// Remove the specified dimensions of size 1 from the shape of a tensor. If no dimensions are
    /// specified, then all dimensions of size 1 will be removed.
    @inlinable
    @differentiable(wrt: self where Scalar : TensorFlowFloatingPoint)
    func squeezingShape(at axes: Int...) -> Tensor {
        return squeezingShape(at: axes)
    }

    /// Remove the specified dimensions of size 1 from the shape of a tensor. If no dimensions are
    /// specified, then all dimensions of size 1 will be removed.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpSqueezingShape(at:) where Scalar : TensorFlowFloatingPoint)
    func squeezingShape(at axes: [Int]) -> Tensor {
        return Raw.squeeze(self, squeezeDims: axes.map(Int32.init))
    }
}

internal extension Tensor where Scalar : TensorFlowFloatingPoint {
    @inlinable
    func _vjpReshaped(toShape newShape: Tensor<Int32>) -> (Tensor, (Tensor) -> Tensor) {
        let value = reshaped(toShape: newShape)
        return (value, { [shape = shapeTensor] v in v.reshaped(toShape: shape) })
    }

    @inlinable
    func _vjpExpandingShape(at shapeIndex: Int) -> (Tensor, (Tensor) -> Tensor) {
        let value = expandingShape(at: shapeIndex)
        return (value, { v in v.squeezingShape(at: shapeIndex) })
    }

    @inlinable
    func _vjpSqueezingShape(at axes: [Int]) -> (Tensor, (Tensor) -> Tensor) {
        let value = squeezingShape(at: axes)
        return (value, { [shape = shapeTensor] v in v.reshaped(toShape: shape) })
    }
}
